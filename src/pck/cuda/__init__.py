"""CUDA kernels. Requires the ``gpu`` extra (PyTorch with CUDA) and nvcc."""

from pck.cuda.checker import (
    DEFAULT_BLOCK,
    CheckResult,
    GpuBlockedChecker,
    PendingCheck,
    SideStreamChecker,
    check_many,
)
from pck.cuda.gemv import (
    gemv_coalesced,
    gemv_column,
    gemv_fp8,
    gemv_naive,
    gemv_template,
    quantize_fp8,
    row_q8,
)
from pck.cuda.sampling import (
    SamplingCheck,
    check_topk_gpu,
    check_topp_gpu,
    sharded_topk_gpu,
    topk_sample,
    topp_select,
)

__all__ = [
    "DEFAULT_BLOCK",
    "CheckResult",
    "GpuBlockedChecker",
    "PendingCheck",
    "SamplingCheck",
    "SideStreamChecker",
    "check_many",
    "check_topk_gpu",
    "check_topp_gpu",
    "gemv_coalesced",
    "gemv_column",
    "gemv_fp8",
    "gemv_naive",
    "gemv_template",
    "quantize_fp8",
    "row_q8",
    "sharded_topk_gpu",
    "topk_sample",
    "topp_select",
]
