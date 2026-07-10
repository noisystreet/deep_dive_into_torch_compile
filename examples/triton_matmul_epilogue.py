"""MatMul + Bias + ReLU 融合 kernel 示例

展示如何将矩阵乘法、bias 加法和 ReLU 激活融合为单个 Triton kernel。
融合避免了中间结果写回全局内存，减少了内存带宽消耗。
"""
# --- docs: matmul_epilogue ---

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bias_relu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """矩阵乘法 (A @ B) + Bias + ReLU 融合 kernel。

    参数:
        a_ptr: A 矩阵指针，形状 (M, K)
        b_ptr: B 矩阵指针，形状 (K, N)
        bias_ptr: bias 向量指针，形状 (N,)
        c_ptr: 输出矩阵指针，形状 (M, N)
        M, N, K: 矩阵维度
        stride_am, stride_ak: A 矩阵的行/列步幅
        stride_bk, stride_bn: B 矩阵的行/列步幅
        stride_cm, stride_cn: C 矩阵的行/列步幅
        BLOCK_SIZE: 分块大小
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE

    # 块索引
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_SIZE)

    # Mask
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # 累加器
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE):
        k_current = k_start + k_offsets
        mask_k = k_current < K

        # 加载 A 的块
        a_ptrs = a_ptr + (
            m_offsets[:, None] * stride_am + k_current[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
        )

        # 加载 B 的块
        b_ptrs = b_ptr + (
            k_current[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
        )

        acc = tl.dot(a, b, acc)

    # --- Epilogue 融合 ---
    # 加载 bias 并广播到块的所有行
    bias = tl.load(bias_ptr + n_offsets, mask=mask_n)
    acc = acc + bias[None, :]

    # ReLU 激活
    acc = tl.where(acc > 0, acc, 0.0)

    # 存储最终结果
    c_ptrs = c_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def matmul_bias_relu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    BLOCK_SIZE: int = 64,
) -> torch.Tensor:
    """包装函数: 执行 MatMul + Bias + ReLU。

    BLOCK_SIZE 默认 64 以避免 shared memory 超限(大部分 GPU 限制 96KB)。
    """

    assert a.is_cuda and b.is_cuda and bias.is_cuda
    M, K = a.shape
    _, N = b.shape
    assert a.shape[1] == b.shape[0]
    assert bias.shape[0] == N

    c = torch.empty((M, N), device="cuda", dtype=a.dtype)

    grid = (
        triton.cdiv(M, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE),
    )

    matmul_bias_relu_kernel[grid](
        a,
        b,
        bias,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c


# --- docs: end ---

if __name__ == "__main__":
    torch.manual_seed(42)
    M, N, K = 1024, 1024, 1024

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((N,), device="cuda", dtype=torch.float16)

    # Triton 实现
    c_triton = matmul_bias_relu(a, b, bias)

    # PyTorch 参考实现
    c_ref = torch.relu(a @ b + bias)

    # 验证正确性
    torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
    print(f"✓ MatMul+Bias+ReLU 融合 kernel 验证通过")
    print(f"  输入形状: A=({M}, {K}), B=({K}, {N}), bias=({N},)")
    print(f"  输出形状: ({M}, {N})")
    print(f"  输出前 5 个元素: {c_triton.flatten()[:5].tolist()}")
