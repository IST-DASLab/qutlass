#
# Test case for bug fix: fusedQuantizeMx scale tensor dimension correctness
#
# Bug: fusedQuantizeMx() allocated scale tensor with padded dimensions instead of actual dimensions
# Fix: Use actual dimensions (M, K//32) instead of padded dimensions
#

import pytest
import torch
from scipy.linalg import hadamard

import qutlass
from qutlass import fusedQuantizeMx


if not torch.cuda.is_available():
    pytest.skip("CUDA required for these tests.", allow_module_level=True)


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")
ROT_SIZE = 32  # Using smallest rotation size for faster testing


class TestScaleTensorDimensions:
    """Test that fusedQuantizeMx produces correct scale tensor dimensions."""

    @pytest.mark.parametrize("method", ["quest", "abs_max"])
    def test_small_row_count(self, method):
        """
        Test the specific bug case: small number of rows (< 128).

        Bug: For input [4, 128], buggy code allocated [128, 4] instead of [4, 4]
        Expected: Scale tensor should have shape [M, K//32] where M=4, K=128
        """
        h = get_hadamard_matrix(ROT_SIZE, DTYPE, DEVICE)

        # Input with 4 rows (less than padding threshold of 128)
        m, k = 4, 128
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE) * 25.0

        # Quantize
        _, a_e8m0 = fusedQuantizeMx(a, h, method=method)

        # Expected shape: [M, K//32] = [4, 128//32] = [4, 4]
        expected_shape = (m, k // 32)

        assert a_e8m0.shape == expected_shape, (
            f"Scale tensor has wrong shape. Expected {expected_shape}, got {a_e8m0.shape}. "
            f"Bug: might be using padded dimensions instead of actual dimensions."
        )

        # Verify all rows are filled (not zeros)
        # Each scale should be non-zero since input is random non-zero values
        # Note: Float8_e8m0fnu doesn't support .abs(), so convert to float first
        non_zero_rows = (a_e8m0.to(torch.float32).abs() > 0).any(dim=1).sum().item()
        assert non_zero_rows == m, (
            f"Only {non_zero_rows}/{m} rows of scale tensor are non-zero! "
            f"Bug: CUDA kernel may not be filling all allocated rows."
        )

    @pytest.mark.parametrize("method", ["quest", "abs_max"])
    @pytest.mark.parametrize("m,k", [
        (1, 128),      # Single row
        (4, 256),      # Small matrix
        (8, 512),      # Small matrix
        (16, 1024),    # Medium matrix
        (127, 2048),   # Just below padding threshold
        (128, 4096),   # Exactly at padding threshold
        (256, 4096),   # Above padding threshold
    ])
    def test_various_shapes(self, method, m, k):
        """Test correct scale tensor dimensions for various input shapes."""
        h = get_hadamard_matrix(ROT_SIZE, DTYPE, DEVICE)
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE) * 25.0

        # Quantize
        _, a_e8m0 = fusedQuantizeMx(a, h, method=method)

        # Expected shape: [M, K//32]
        # Note: MX format uses 32-element blocks, not 16!
        expected_shape = (m, k // 32)

        assert a_e8m0.shape == expected_shape, (
            f"For input shape {a.shape}, scale tensor should be {expected_shape}, "
            f"but got {a_e8m0.shape}"
        )

        # Verify all rows contain non-zero values
        # Note: Float8_e8m0fnu doesn't support .abs(), so convert to float first
        non_zero_rows = (a_e8m0.to(torch.float32).abs() > 0).any(dim=1).sum().item()
        assert non_zero_rows == m, (
            f"Input has {m} rows, but only {non_zero_rows} rows in scale tensor are non-zero"
        )

    @pytest.mark.parametrize("method", ["quest", "abs_max"])
    def test_batched_input(self, method):
        """Test correct dimensions for batched inputs."""
        h = get_hadamard_matrix(ROT_SIZE, DTYPE, DEVICE)

        # Batched input: [batch_size, M, K]
        batch_size, m, k = 2, 4, 128
        a = torch.randn(batch_size, m, k, dtype=DTYPE, device=DEVICE) * 25.0

        # Quantize
        _, a_e8m0 = fusedQuantizeMx(a, h, method=method)

        # For batched input [B, M, K], scale flattens batch and row dims: [B*M, K//32]
        # This handles batching correctly using a.numel() // a.size(-1)
        expected_rows = batch_size * m
        expected_cols = k // 32
        expected_shape = (expected_rows, expected_cols)

        assert a_e8m0.shape == expected_shape, (
            f"For batched input shape {a.shape}, scale tensor should be {expected_shape}, "
            f"but got {a_e8m0.shape}"
        )

        # Verify all rows are filled
        # Note: Float8_e8m0fnu doesn't support .abs(), so convert to float first
        non_zero_rows = (a_e8m0.to(torch.float32).abs() > 0).any(dim=1).sum().item()
        assert non_zero_rows == expected_rows, (
            f"Only {non_zero_rows}/{expected_rows} rows in scale tensor are non-zero"
        )

    @pytest.mark.parametrize("method", ["quest", "abs_max"])
    def test_block_size_is_32_not_16(self, method):
        """
        Verify that MX format uses 32-element blocks, not 16.

        This is a regression test for the bug fix - the fix initially used
        block_size=16, but MX format actually uses block_size=32.
        """
        h = get_hadamard_matrix(ROT_SIZE, DTYPE, DEVICE)

        m, k = 8, 1024
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE) * 25.0

        _, a_e8m0 = fusedQuantizeMx(a, h, method=method)

        # If using block_size=16 (WRONG), shape would be [8, 1024//16] = [8, 64]
        wrong_shape_with_bs16 = (m, k // 16)

        # If using block_size=32 (CORRECT), shape would be [8, 1024//32] = [8, 32]
        correct_shape_with_bs32 = (m, k // 32)

        assert a_e8m0.shape != wrong_shape_with_bs16, (
            f"Scale tensor has shape {a_e8m0.shape} which suggests block_size=16. "
            f"MX format should use block_size=32!"
        )

        assert a_e8m0.shape == correct_shape_with_bs32, (
            f"Scale tensor should have shape {correct_shape_with_bs32} (block_size=32), "
            f"but got {a_e8m0.shape}"
        )

    @pytest.mark.parametrize("method", ["quest", "abs_max"])
    def test_no_extra_padding_in_scale_tensor(self, method):
        """
        Verify that scale tensor is NOT allocated with padded dimensions.

        Bug: Original code used get_padded_shape_mx() which pads rows to multiples of 128
        Fix: Use actual dimensions, padding only happens later in to_blocked()
        """
        h = get_hadamard_matrix(ROT_SIZE, DTYPE, DEVICE)

        # Use 4 rows (which would be padded to 128 with get_padded_shape_mx)
        m, k = 4, 128
        a = torch.randn(m, k, dtype=DTYPE, device=DEVICE) * 25.0

        _, a_e8m0 = fusedQuantizeMx(a, h, method=method)

        # Buggy code would allocate [128, 4] (padded_rows=128, cols=128//32=4)
        buggy_padded_shape = (128, k // 32)

        # Fixed code should allocate [4, 4] (actual_rows=4, cols=128//32=4)
        correct_shape = (m, k // 32)

        assert a_e8m0.shape != buggy_padded_shape, (
            f"Scale tensor has padded shape {buggy_padded_shape}! "
            f"Should use actual dimensions {correct_shape}."
        )

        assert a_e8m0.shape == correct_shape, (
            f"Scale tensor should have actual dimensions {correct_shape}, "
            f"not padded dimensions. Got {a_e8m0.shape}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
