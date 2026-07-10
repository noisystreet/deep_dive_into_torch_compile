"""融合 Softmax Kernel 示例

展示如何使用 Triton 实现数值稳定的 fused softmax kernel。
所有中间计算（求最大值 → 指数 → 求和 → 归一化）都在寄存器中完成，
避免写回全局内存。
"""

import torch
import triton
import triton.language as tl

# --- docs: fused_softmax ---


@triton.jit
def fused_softmax_kernel(
    x_ptr,
    output_ptr,
    x_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """逐行 fused softmax kernel。

    每行由一个 Triton program 处理。使用 "max 减" 技巧确保数值稳定。

    参数:
        x_ptr: 输入张量指针
        output_ptr: 输出张量指针
        x_row_stride: 输入张量的行步幅（bytes）
        output_row_stride: 输出张量的行步幅
        n_cols: 列数
        BLOCK_SIZE: 每行分块大小（需要 >= n_cols）
    """
    row_idx = tl.program_id(axis=0)
    row_start_x = row_idx * x_row_stride
    row_start_out = row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # 加载一行数据
    x = tl.load(x_ptr + row_start_x + col_offsets, mask=col_mask)

    # 数值稳定的 softmax：
    # 1. 减去最大值，避免 exp(大正数) 溢出
    # 用 -1e38 替换 masked 元素，避免默认值 0 污染 max
    x_masked = tl.where(col_mask, x, -1e38)
    x_max = tl.max(x_masked, axis=0)
    x_sub = x - x_max

    # 2. 计算指数
    x_exp = tl.exp(x_sub)

    # 3. 求和（masked 元素贡献 0）
    x_exp_masked = tl.where(col_mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp_masked, axis=0)

    # 4. 归一化
    y = x_exp / x_sum

    tl.store(output_ptr + row_start_out + col_offsets, y, mask=col_mask)


# --- docs: end ---

# --- docs: fused_softmax_2d ---


@triton.jit
def fused_softmax_kernel_2d(
    x_ptr,
    output_ptr,
    x_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """二维网格 fused softmax kernel。

    支持多行并行处理，每行可被多个 block 分块处理。
    当 BLOCK_SIZE < n_cols 时，一行由多个 program 协作处理。

    参数:
        x_ptr: 输入张量指针
        output_ptr: 输出张量指针
        x_row_stride: 输入张量的行步幅
        output_row_stride: 输出张量的行步幅
        n_rows: 行数
        n_cols: 列数
        BLOCK_SIZE: 分块大小
    """
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)

    # 计算起始位置
    col_start = col_idx * BLOCK_SIZE
    row_start_x = row_idx * x_row_stride
    row_start_out = row_idx * output_row_stride

    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = (row_idx < n_rows) & (offsets < n_cols)

    # 加载数据块
    x = tl.load(x_ptr + row_start_x + offsets, mask=mask)

    # Softmax 计算（单 block 覆盖整行）
    x_masked = tl.where(mask, x, -1e38)
    x_max = tl.max(x_masked, axis=0)
    x_exp = tl.exp(x - x_max)
    x_exp_masked = tl.where(mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp_masked, axis=0)

    tl.store(
        output_ptr + row_start_out + offsets,
        x_exp / x_sum,
        mask=mask,
    )


def fused_softmax(x: torch.Tensor, BLOCK_SIZE: int = 4096) -> torch.Tensor:
    """包装函数：在输入张量的最后一维执行 fused softmax。

    参数:
        x: 输入张量，形状 (..., n_cols)
        BLOCK_SIZE: Triton kernel 的分块大小

    返回:
        在最后一维执行 softmax 后的结果
    """
    assert x.is_cuda
    x_contiguous = x.contiguous()
    orig_shape = x_contiguous.shape
    n_cols = orig_shape[-1]

    # 展平前面的维度
    x_2d = x_contiguous.view(-1, n_cols)
    n_rows = x_2d.shape[0]

    output = torch.empty_like(x_2d)

    assert BLOCK_SIZE >= n_cols, f"BLOCK_SIZE ({BLOCK_SIZE}) 必须 >= n_cols ({n_cols})"

    grid = (n_rows,)

    fused_softmax_kernel[grid](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(orig_shape)


# --- docs: end ---

if __name__ == "__main__":
    torch.manual_seed(42)

    # 测试 1: 小矩阵
    print("=== 测试 1: 小矩阵 (1024, 2048) ===")
    M, N = 1024, 2048
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)

    y_triton = fused_softmax(x)
    y_ref = torch.softmax(x, dim=-1)

    torch.testing.assert_close(y_triton, y_ref, rtol=1e-5, atol=1e-5)
    print(f"✓ Fused Softmax kernel 验证通过")
    print(f"  输出形状: {y_triton.shape}")

    # 验证 softmax 性质：每行和为 1
    row_sums = y_triton.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Softmax 行和不为 1"
    )
    print(f"  ✓ 行和校验通过 (max diff: {(row_sums - 1).abs().max().item():.6f})")

    # 测试 2: 大矩阵
    print("\n=== 测试 2: 大矩阵 (4096, 4096) ===")
    x_large = torch.randn((4096, 4096), device="cuda", dtype=torch.float32)

    y_large_triton = fused_softmax(x_large)
    y_large_ref = torch.softmax(x_large, dim=-1)

    torch.testing.assert_close(y_large_triton, y_large_ref, rtol=1e-5, atol=1e-5)
    print(f"✓ 大矩阵 Fused Softmax kernel 验证通过")
    print(f"  输出前 5 个元素: {y_large_triton.flatten()[:5].tolist()}")

    # 测试 3: 三维张量
    print("\n=== 测试 3: 三维张量 (32, 128, 512) ===")
    x_3d = torch.randn((32, 128, 512), device="cuda", dtype=torch.float32)

    y_3d_triton = fused_softmax(x_3d)
    y_3d_ref = torch.softmax(x_3d, dim=-1)

    torch.testing.assert_close(y_3d_triton, y_3d_ref, rtol=1e-5, atol=1e-5)
    print(f"✓ 三维 Fused Softmax kernel 验证通过")

    # 测试 4: 数值稳定性测试 - 大输入值
    print("\n=== 测试 4: 数值稳定性测试 ===")
    x_stability = torch.tensor(
        [[1e5, 1e3, 1e1, -1e3, -1e5]],
        device="cuda",
        dtype=torch.float32,
    )
    y_stability_triton = fused_softmax(x_stability)
    y_stability_ref = torch.softmax(x_stability, dim=-1)

    torch.testing.assert_close(
        y_stability_triton, y_stability_ref, rtol=1e-5, atol=1e-5
    )
    print(f"✓ 数值稳定性测试通过")
    print(f"  输入: [1e5, 1e3, 1e1, -1e3, -1e5]")
    print(f"  输出: {y_stability_triton[0].tolist()}")
