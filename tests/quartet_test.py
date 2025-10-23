import torch

from scipy.linalg import hadamard

def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )

from qutlass import (
    matmul_mxf4_bf16_tn,
    matmul_mxf8_bf16_tn,
    fusedQuantizeMx,
    backward_t_bf16,
    backward_qt_bf16,
)
from qutlass.utils import to_blocked


def _rtne_fp4(x: torch.Tensor):
    device = x.device
    grid = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=x.dtype,
        device=x.device,
    )
    grid_int = torch.tensor(
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=device,
    )
    inds = torch.bucketize(x, grid)
    lo, hi = (inds - 1).clamp(min=0, max=15), inds.clamp(min=0, max=15)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & (grid_int[hi] % 2 == 0)
    y = torch.where(pick_hi, g_hi, g_lo)
    y_int = torch.where(pick_hi, grid_int[hi], grid_int[lo])
    y_int_packed = (y_int[..., 1::2] & 0xF) << 4 | y_int[..., ::2] & 0xF
    return y, y_int_packed


def _dq_fp4(x_e2m1: torch.Tensor, x_e8m0: torch.Tensor, alpha: float):
    device = x_e2m1.device

    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]

    scales_dq = x_e8m0.to(torch.float64)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 32)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) / alpha  # * (4. / 3.)

    return x_dq, x_fp4_dq, scales_dq


def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(
        *clip_mask.shape[:-1],
        clip_mask.size(-1) * 8,
        dtype=torch.bool,
        device=clip_mask.device,
    )
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


def _forward_quantize_ref(x: torch.Tensor, h: torch.Tensor):
    device = x.device

    xh_ref64 = (
        x.unflatten(dim=-1, sizes=(-1, 32)).to(dtype=torch.float64)
        @ h.reshape(32, 32).to(dtype=torch.float64)
    ).flatten(start_dim=-2)
    scales_ref64_ = (
        xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).std(dim=-1, correction=0)
        * (2.92247856 / 6.0)
        + 1e-8
    )
    xh_e8m0_ref = scales_ref64_.log2().floor().exp2().to(dtype=torch.float8_e8m0fnu)
    scales_ref64 = xh_e8m0_ref.to(dtype=torch.float64)
    xh_scaled_ref64 = (
        xh_ref64.unflatten(dim=-1, sizes=(-1, 32)) / scales_ref64[..., None]
    ).flatten(start_dim=-2)

    clip_mask_unpacked_ref = xh_scaled_ref64.abs() < 6.0
    clip_mask_ref = torch.zeros(
        *x.shape[:-1], x.size(-1) // 8, dtype=torch.uint8, device=device
    )
    for i in range(8):
        clip_mask_ref |= clip_mask_unpacked_ref[..., i::8].to(dtype=torch.uint8) << i

    xh_fp4_ref, xh_e2m1_ref = _rtne_fp4(xh_scaled_ref64)

    xh_dq, xh_fp4_dq, scales_dq = _dq_fp4(xh_e2m1_ref, xh_e8m0_ref, alpha=1.0)
    clip_mask_unpacked_dq = _unpack_mask(clip_mask_ref)
    assert xh_fp4_dq.equal(xh_fp4_ref)
    assert scales_dq.equal(scales_ref64)
    assert clip_mask_unpacked_dq.equal(clip_mask_unpacked_ref)

    return xh_dq, clip_mask_unpacked_ref, (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref)


def _backward_quantize_ref(x: torch.Tensor, h: torch.Tensor):
    xh_ref64 = (
        x.unflatten(dim=-1, sizes=(-1, 32)).to(dtype=torch.float64)
        @ h.reshape(32, 32).to(dtype=torch.float64)
    ).flatten(start_dim=-2)
    scales_ref64_ = xh_ref64.unflatten(dim=-1, sizes=(-1, 32)).abs().amax(dim=-1)
    xh_e8m0_ref = scales_ref64_.log2().floor().exp2().to(dtype=torch.float8_e8m0fnu)
    scales_ref64 = xh_e8m0_ref.to(dtype=torch.float64)
    xh_scaled_ref64 = (
        xh_ref64.unflatten(dim=-1, sizes=(-1, 32)) / scales_ref64[..., None]
    ).flatten(start_dim=-2) * 3.0  # * .75

    xh_fp4_ref, xh_e2m1_ref = _rtne_fp4(xh_scaled_ref64)

    xh_dq, x_fp4_dq, scales_dq = _dq_fp4(xh_e2m1_ref, xh_e8m0_ref, alpha=3.0)
    assert x_fp4_dq.equal(xh_fp4_ref)
    assert scales_dq.equal(scales_ref64)

    return xh_dq, (xh_e2m1_ref, xh_e8m0_ref)


def _unit_test() -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda")

    l, n, k = 2, 4096, 4096
    hadamard_matrix = get_hadamard_matrix(32, dtype=dtype, device="cuda")
    x = torch.randn(l, n, k, dtype=dtype, device=device) * 25.0

    xh_dq_ref, clip_mask_unpacked_ref, (xh_e2m1_ref, xh_e8m0_ref, clip_mask_ref) = (
        _forward_quantize_ref(x, hadamard_matrix)
    )
    xh_e2m1, xh_e8m0, clip_mask = fusedQuantizeMx(
        x, hadamard_matrix, method="quest", return_mask=True
    )
    xh_e8m0 = xh_e8m0.reshape(l, n, k // 32)
    assert xh_e8m0.equal(xh_e8m0_ref)
    assert clip_mask.equal(clip_mask_ref)
    xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=1.0)
    assert xh_dq.equal(xh_dq_ref)
    diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    print(diff, diff / xh_e2m1.numel())
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    xh_e2m1, xh_e8m0, clip_mask = fusedQuantizeMx(
        x, hadamard_matrix, method="quest", return_mask=True
    )
    diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    print(diff, diff / xh_e2m1.numel())
    xh_e8m0 = xh_e8m0.reshape(l, n, k // 32)
    xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=1.0)
    assert xh_dq.equal(xh_dq_ref)
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    xh_dq_ref, (xh_e2m1_ref, xh_e8m0_ref) = _backward_quantize_ref(x, hadamard_matrix)
    # xh_e2m1, xh_e8m0 = quartet.backward_bf16(x, hadamard_matrix)
    xh_e2m1, xh_e8m0 = fusedQuantizeMx(x, hadamard_matrix, method="abs_max")
    xh_e8m0 = xh_e8m0.reshape(l, n, k // 32)
    assert xh_e8m0.equal(xh_e8m0_ref)
    # xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0)
    # assert xh_dq.equal(xh_dq_ref)
    diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    print(diff, diff / xh_e2m1.numel())
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    xh_e2m1, xh_e8m0 = fusedQuantizeMx(x, hadamard_matrix, method="abs_max")
    xh_e8m0 = xh_e8m0.reshape(l, n, k // 32)
    assert xh_e8m0.equal(xh_e8m0_ref)
    diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    print(diff, diff / xh_e2m1.numel())
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    xh_dq_ref, (xh_e2m1_ref, xh_e8m0_ref) = _backward_quantize_ref(
        x.transpose(-2, -1), hadamard_matrix
    )
    xh_e2m1, xh_e8m0 = backward_t_bf16(x, hadamard_matrix)
    assert xh_e8m0.equal(xh_e8m0_ref)
    # xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=3.)
    # assert xh_dq.equal(xh_dq_ref)
    diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    print(diff, diff / xh_e2m1.numel())
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    # xh_e2m1_, xh_e8m0_ = quartet.backward_bf16(x, hadamard_matrix)
    xh_e2m1_, xh_e8m0_ = fusedQuantizeMx(x, hadamard_matrix, method="abs_max")
    xh_e8m0_ = xh_e8m0_.reshape(l, n, k // 32)
    xh_dq_ref, (xh_e2m1_ref, xh_e8m0_ref) = _backward_quantize_ref(
        _dq_fp4(xh_e2m1_, xh_e8m0_, alpha=3.0)[0].transpose(-2, -1), hadamard_matrix
    )
    alpha = torch.Tensor([3.0]).to(device)
    xh_e2m1, xh_e8m0 = backward_qt_bf16(
        xh_e2m1_, xh_e8m0_, hadamard_matrix, alpha=alpha
    )
    assert xh_e8m0.equal(xh_e8m0_ref)
    xh_dq, *_ = _dq_fp4(xh_e2m1, xh_e8m0, alpha=3.0)
    assert xh_dq.equal(xh_dq_ref)
    # diff = (xh_e2m1.view(dtype=xh_e2m1_ref.dtype) != xh_e2m1_ref).sum().item()
    # print(diff, diff / xh_e2m1.numel())
    # assert xh_e2m1.view(dtype=xh_e2m1_ref.dtype).equal(xh_e2m1_ref)

    m, n, k = 4096 * 3, 4096 * 2, 4096
    a = torch.randn(m, k, dtype=dtype, device=device) * 25.0
    b = torch.randn(n, k, dtype=dtype, device=device) * 25.0

    a_e2m1, a_e8m0, _ = fusedQuantizeMx(
        a, hadamard_matrix, method="quest", return_mask=True
    )
    b_e2m1, b_e8m0, _ = fusedQuantizeMx(
        b, hadamard_matrix, method="quest", return_mask=True
    )
    a_dq, *_ = _dq_fp4(a_e2m1, a_e8m0, alpha=1.0)
    b_dq, *_ = _dq_fp4(b_e2m1, b_e8m0, alpha=1.0)
    out_ref = a_dq @ b_dq.transpose(-2, -1)
    a_scale_block = to_blocked(a_e8m0)
    b_scale_block = to_blocked(b_e8m0)
    alpha = torch.Tensor([1.0]).to(device)
    out = matmul_mxf4_bf16_tn(a_e2m1, b_e2m1, a_scale_block, b_scale_block, alpha)
    assert out.equal(out_ref.to(dtype=out.dtype))

    print("Passed!")


def _mm_fp8_test() -> None:
    m, n, k = 4096 * 3, 4096, 4096 * 2
    a_e4m3 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    b_e4m3 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    a_e8m0 = torch.ones(m, k // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    b_e8m0 = torch.ones(n, k // 32, dtype=torch.float8_e8m0fnu, device="cuda")
    # a_e8m0_blocked = to_blocked(a_e8m0)
    # b_e8m0_blocked = to_blocked(b_e8m0)
    alpha = torch.Tensor([1.0]).to("cuda")
    out = matmul_mxf8_bf16_tn(a_e4m3, b_e4m3, a_e8m0, b_e8m0, alpha)
    out_ref = (
        a_e4m3.to(dtype=torch.float64)
        @ b_e4m3.transpose(-2, -1).to(dtype=torch.float64)
    ).to(dtype=torch.bfloat16)
    print(((out != out_ref).sum() / out.numel()).item())


if __name__ == "__main__":
    _unit_test()
    _mm_fp8_test()
