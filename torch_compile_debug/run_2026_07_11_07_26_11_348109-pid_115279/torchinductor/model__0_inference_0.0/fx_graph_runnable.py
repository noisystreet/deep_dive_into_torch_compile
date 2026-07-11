import os

os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_gzz"

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims


import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False


isolate_fails_code_str = None


if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache

    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.12.1+cu130
# torch cuda version: 13.0
# torch git version: 7269437d655783a26cba32aa88195b741ff496aa


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Tue_Dec_16_07:23:41_PM_PST_2025
# Cuda compilation tools, release 13.1, V13.1.115
# Build cuda_13.1.r13.1/compiler.37061995_0

# GPU Hardware Info:
# NVIDIA GeForce RTX 4060 Laptop GPU : 1

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg0_1):
        sin = torch.ops.aten.sin.default(arg0_1)
        cos = torch.ops.aten.cos.default(arg0_1)
        arg0_1 = None
        add = torch.ops.aten.add.Tensor(sin, cos)
        sin = cos = None
        sum_1 = torch.ops.aten.sum.default(add)
        add = None
        return (sum_1,)


def load_args(reader):
    buf0 = reader.storage(None, 40)
    reader.tensor(buf0, (10,), is_leaf=True)  # arg0_1


load_args._version = 0
mod = Repro()
if __name__ == "__main__":
    from torch._dynamo.repro.after_aot import run_repro

    with torch.no_grad():
        run_repro(
            mod,
            load_args,
            accuracy=False,
            command="run",
            save_dir=None,
            tracing_mode="real",
            check_str=None,
        )
        # To run it separately, do
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)
