# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_add_cos_sin_sum_0 = async_compile.cpp_pybinding(
    ["const float*", "float*"],
    r"""
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        {
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(10L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = tmp0.sin();
                        auto tmp2 = tmp0.cos();
                        auto tmp3 = tmp1 + tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    if(C10_UNLIKELY(x0 >= static_cast<int64_t>(8L) && x0 < static_cast<int64_t>(10L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(2L));
                        auto tmp1 = tmp0.sin();
                        auto tmp2 = tmp0.cos();
                        auto tmp3 = tmp1 + tmp2;
                        tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp3, static_cast<int64_t>(2L));
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<int64_t>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
}
""",
)


async_compile.wait(globals())
del async_compile


class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        (arg0_1,) = args
        args.clear()
        assert_size_stride(arg0_1, (10,), (1,))
        buf0 = empty_strided_cpu((), (), torch.float32)
        # [Provenance debug handles] cpp_fused_add_cos_sin_sum_0:1
        cpp_fused_add_cos_sin_sum_0(arg0_1, buf0)
        del arg0_1
        return (buf0,)


runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided

    arg0_1 = rand_strided((10,), (1,), device="cpu", dtype=torch.float32)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance

    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    args = get_args()
    compiled_module_main(
        "None",
        lambda times, repeat: benchmark_compiled_module(
            args, times=times, repeat=repeat
        ),
    )
