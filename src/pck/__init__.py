"""Proof-carrying kernels: checksum checkers with proven thresholds."""

from pck.checker import BlockedChecker
from pck.faults import flip_bit
from pck.fp import U32, U64, bf16, ceil_div, gamma, round_up_f32, tree_height
from pck.gemv import WARP, abs_row_sums, emulate_gemv, emulate_gemv_coalesced, reference_gemv

__all__ = [
    "U32",
    "U64",
    "WARP",
    "BlockedChecker",
    "abs_row_sums",
    "bf16",
    "ceil_div",
    "emulate_gemv",
    "emulate_gemv_coalesced",
    "flip_bit",
    "gamma",
    "reference_gemv",
    "round_up_f32",
    "tree_height",
]
