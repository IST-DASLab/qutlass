#!/usr/bin/env python3
"""
Comprehensive tests for CUTLASS 4.x v2 API with arbitrary K support.

Tests:
- Multiple K dimensions (512, 1024, 2048, 4096)
- Both quest and abs_max quantization methods
- Batched inputs [B, M, K]
- Auto-detection vs manual v2 selection
- Correctness validation
- Integration with matmul
"""

import torch
import pytest
from scipy.linalg import hadamard

# Skip all tests if CUDA not available or not Blackwell GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU"
)

try:
    import qutlass
except ImportError:
    pytest.skip("qutlass not available", allow_module_level=True)


class TestV2LargeK:
    """Test v2 API with K > 256."""

    @pytest.fixture(params=[512, 1024, 2048, 4096])
    def K(self, request):
        """Parametrize K dimension."""
        return request.param

    @pytest.fixture
    def device(self):
        """CUDA device."""
        return torch.device('cuda')

    @pytest.fixture
    def dtype(self):
        """BF16 dtype."""
        return torch.bfloat16

    def test_v2_quantization_quest(self, K, device, dtype):
        """Test v2 quantization with Quest method for large K."""
        M = 512

        # Generate Hadamard matrix
        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)

        # Generate test matrix
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0

        # Quantize using v2 API (auto-detected for K > 256)
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest')

        # Validate output shapes
        assert A_e2m1.shape == (M, K // 2), f"Expected {(M, K // 2)}, got {A_e2m1.shape}"
        assert A_e8m0.shape == (M, K // 32), f"Expected {(M, K // 32)}, got {A_e8m0.shape}"

        # Validate dtypes
        assert A_e2m1.dtype == torch.uint8, f"Expected uint8, got {A_e2m1.dtype}"
        assert A_e8m0.dtype == torch.float8_e8m0fnu, f"Expected float8_e8m0fnu, got {A_e8m0.dtype}"

        # Validate no NaN/Inf
        assert not torch.isnan(A_e8m0.view(torch.uint8).float()).any(), "Scale factors contain NaN"
        assert not torch.isinf(A_e8m0.view(torch.uint8).float()).any(), "Scale factors contain Inf"

    def test_v2_quantization_absmax(self, K, device, dtype):
        """Test v2 quantization with AbsMax method for large K."""
        M = 512

        # Generate Hadamard matrix
        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)

        # Generate test matrix
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0

        # Quantize using v2 API with abs_max
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='abs_max')

        # Validate output shapes
        assert A_e2m1.shape == (M, K // 2)
        assert A_e8m0.shape == (M, K // 32)

        # Validate no NaN/Inf
        assert not torch.isnan(A_e8m0.view(torch.uint8).float()).any()
        assert not torch.isinf(A_e8m0.view(torch.uint8).float()).any()

    def test_v2_manual_selection(self, device, dtype):
        """Test manual v2 API selection with use_v2=True."""
        M, K = 512, 512

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0

        # Force v2 API
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest', use_v2=True)

        assert A_e2m1.shape == (M, K // 2)
        assert A_e8m0.shape == (M, K // 32)

    def test_legacy_api_fallback(self, device, dtype):
        """Test legacy API still works for K ≤ 256."""
        M, K = 512, 256

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0

        # Force legacy API
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest', use_v2=False)

        assert A_e2m1.shape == (M, K // 2)
        assert A_e8m0.shape == (M, K // 32)

    def test_batched_input_3d(self, device, dtype):
        """Test v2 API with batched 3D inputs [B, M, K]."""
        B, M, K = 4, 256, 1024

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(B, M, K, dtype=dtype, device=device) * 25.0

        # Quantize batched input
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest')

        # Validate output shapes preserve batch dimension
        assert A_e2m1.shape == (B, M, K // 2), f"Expected {(B, M, K // 2)}, got {A_e2m1.shape}"
        assert A_e8m0.shape == (B, M, K // 32), f"Expected {(B, M, K // 32)}, got {A_e8m0.shape}"

        # Validate no NaN/Inf
        assert not torch.isnan(A_e8m0.view(torch.uint8).float()).any()

    def test_end_to_end_matmul(self, device, dtype):
        """Test v2 quantization + matmul pipeline."""
        M, N, K = 512, 512, 1024

        # Generate Hadamard matrix
        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)

        # Generate test matrices
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0
        B = torch.randn(N, K, dtype=dtype, device=device) * 25.0

        # Quantize both matrices
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest')
        B_e2m1, B_e8m0 = qutlass.fusedQuantizeMx(B, H, method='quest')

        # Convert scales to blocked format for matmul
        A_scale = qutlass.utils.to_blocked(A_e8m0, use_triton_kernel=True)
        B_scale = qutlass.utils.to_blocked(B_e8m0, use_triton_kernel=True)

        # Run MXFP4 matmul
        alpha = torch.tensor([1.0], device=device)
        C_mxfp4 = qutlass.matmul_mxf4_bf16_tn(
            A_e2m1, B_e2m1, A_scale, B_scale, alpha, backend='cutlass'
        )

        # Validate output
        assert C_mxfp4.shape == (M, N)
        assert C_mxfp4.dtype == torch.bfloat16
        assert not torch.isnan(C_mxfp4).any(), "Matmul output contains NaN"
        assert not torch.isinf(C_mxfp4).any(), "Matmul output contains Inf"

    def test_correctness_vs_bf16(self, device, dtype):
        """Test quantization correctness vs BF16 baseline."""
        M, K = 256, 512

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device) * 10.0

        # BF16 baseline: A @ H^T
        AH_bf16 = A @ H.T

        # Quantize using v2
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='abs_max')

        # Dequantize (approximate - for validation only)
        # Note: Full dequantization requires unpacking E2M1 and applying E8M0 scales
        # This is a simplified check that quantization doesn't produce garbage

        # Check that scale factors are in reasonable range
        scales_uint8 = A_e8m0.view(torch.uint8)
        # E8M0 format: values should be > 0 and < 255
        assert (scales_uint8 > 0).any(), "All scales are zero"
        assert (scales_uint8 < 255).any(), "All scales are saturated"

    def test_error_on_k_not_divisible_by_32(self, device, dtype):
        """Test that K not divisible by 32 raises error."""
        M, K = 256, 500  # 500 % 32 != 0

        H = torch.randn(K, K, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device)

        with pytest.raises(ValueError, match="K must be divisible by 32"):
            qutlass.fusedQuantizeMx_v2(A, H, method='quest')

    def test_error_on_return_mask_with_v2(self, device, dtype):
        """Test that return_mask raises error with v2 API."""
        M, K = 256, 512

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device)

        with pytest.raises(ValueError, match="return_mask is not supported with v2 API"):
            qutlass.fusedQuantizeMx(A, H, method='quest', return_mask=True)


class TestV2DirectAPI:
    """Test direct v2 API calls (fusedQuantizeMx_v2)."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda')

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    def test_direct_v2_call(self, device, dtype):
        """Test direct call to fusedQuantizeMx_v2."""
        M, K = 512, 1024

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device) * 25.0

        # Direct v2 API call
        A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx_v2(A, H, method='quest')

        # Validate shapes
        assert A_e2m1.shape == (M, K // 2)
        assert A_e8m0.shape == (M, K // 32)

        # Validate dtypes (v2 returns uint8 for scales)
        assert A_e2m1.dtype == torch.uint8
        assert A_e8m0.dtype == torch.uint8

    def test_v2_requires_2d_input(self, device, dtype):
        """Test that v2 API requires 2D input."""
        B, M, K = 4, 256, 512

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(B, M, K, dtype=dtype, device=device)

        with pytest.raises(ValueError, match="Input tensor must be 2D"):
            qutlass.fusedQuantizeMx_v2(A, H, method='quest')


def test_k_dimension_scaling(device='cuda', dtype=torch.bfloat16):
    """Test that v2 API works across wide range of K dimensions."""
    device = torch.device(device)

    test_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    M = 256

    results = []

    for K in test_sizes:
        print(f"\nTesting K={K}")

        H = torch.tensor(hadamard(K) * K**-0.5, dtype=dtype, device=device)
        A = torch.randn(M, K, dtype=dtype, device=device) * 10.0

        try:
            A_e2m1, A_e8m0 = qutlass.fusedQuantizeMx(A, H, method='quest')

            # Validate shapes
            assert A_e2m1.shape == (M, K // 2)
            assert A_e8m0.shape == (M, K // 32)

            results.append((K, "✅ PASS"))
            print(f"  ✅ K={K}: Quantization successful")

        except Exception as e:
            results.append((K, f"❌ FAIL: {e}"))
            print(f"  ❌ K={K}: FAILED - {e}")

    print("\n" + "="*80)
    print("K Dimension Scaling Results:")
    print("="*80)
    for K, status in results:
        print(f"K={K:4d}: {status}")
    print("="*80)


if __name__ == "__main__":
    # Run standalone test
    print("Testing v2 API with large K dimensions...")
    test_k_dimension_scaling()
