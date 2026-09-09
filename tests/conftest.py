# -*- coding: utf-8 -*-
"""Test-suite-wide setup.

Blocks mamba-ssm / causal-conv1d from being importable *before* transformers'
Mamba2 modeling module is first loaded (by ``model.mimir_mamba2``, imported
from ``test_mimir_mamba2.py``). transformers picks its fused Triton/CUDA
kernels for Mamba2 purely from whether those two packages import
successfully — not from where any given tensor actually lives (see
``modeling_mamba2.py``'s ``use_kernel_func_from_hub_with_fallback``) — and
bakes that choice in once, at module-import time, as the literal function
objects ``Mamba2Mixer.forward`` calls thereafter.

Left unblocked, a box that happens to have both packages installed *and* a
GPU (this repo's normal training box) runs these tests through the fused
kernels, which additionally require production-scale tensor shapes
(state_size, head_dim, chunk_size all above some minimum) that the
deliberately tiny toy configs in test_mimir_mamba2.py don't meet — e.g.
"causal_conv1d with channel last layout requires strides ... to be multiples
of 8", or (moving the toy model to CPU without moving the toy inputs, or vice
versa) "Expected x.is_cuda() to be true, but got false." Blocking the
packages up front forces the pure-PyTorch scan (numerically correct — see
mimir_mamba2._warn_if_slow_scan_path) unconditionally, which is what these
tests were written against and what actually runs in CI (no GPU, packages
not installed), so the suite behaves identically on every machine instead of
only where mamba-ssm/causal-conv1d happen not to be installed.
"""

from __future__ import annotations

import sys

for _pkg in ("mamba_ssm", "causal_conv1d"):
    sys.modules[_pkg] = None  # type: ignore[assignment]
