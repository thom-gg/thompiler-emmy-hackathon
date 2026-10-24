"""Handlers for shared memory and barriers (and, later, async copies)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.kernel.ir import Smem, Sync

if TYPE_CHECKING:
    from thompiler.walk import Walker

# Smem.dtype is a C type name. [OBS] only "float" is used.
SMEM_DTYPES: dict[str, tuple[str, int]] = {"float": ("f32", 4)}   # C type -> (dtype, bytes)

# [DOC] PROVENANCE.md: shared arrays up to 48 KiB are static; above that emmy
# uses one dynamic pool sized by meta.json's smem_bytes ([OBS] only m01).
STATIC_SMEM_LIMIT = 48 * 1024


@dataclass
class SharedArray:
    name: str                   # IR name, e.g. "acc0_smem"
    symbol: str                 # PTX symbol of the .shared declaration
    extents: tuple[int, ...]    # used to flatten Load / Write indices
    dtype: str                  # "f32"
    nbytes: int                 # size of one element
    size_bytes: int             # size of the whole array: prod(extents) * nbytes


def emit_smem(w: Walker, smem: Smem) -> None:
    """Declare one shared array for the block, and record it so that Load /
    Write on its name use shared memory.

    [EMMY] Smem docstring: renders to `__shared__ [__align__(N)] <dtype>
    name[prod(extents)]`; extents flatten Load / Write indices; align = 0
    means the dtype's natural alignment.
    The declaration is module-level, like examples/k_sum_1_reduce.ptx; the
    symbol gets an "s_" prefix so an IR name never collides with a label.
    """
    if smem.dtype not in SMEM_DTYPES:
        raise NotImplementedError(f"Smem {smem.name} of type {smem.dtype!r}")
    dtype, nbytes = SMEM_DTYPES[smem.dtype]
    extents = tuple(int(e) for e in smem.extents)

    known = w.smem.get(smem.name)
    if known is not None:
        if known.extents != extents or known.dtype != dtype:
            raise ValueError(f"Smem {smem.name} declared twice with different shapes")
        return

    array_bytes = prod(extents) * nbytes

    # Check that the size of all already declared smem arrays + this new one
    # doesnt exceed the static limit
    declared_bytes = sum(a.size_bytes for a in w.smem.values())
    total_bytes = declared_bytes + array_bytes
    if total_bytes > STATIC_SMEM_LIMIT:
        raise NotImplementedError(
            f"{total_bytes} bytes of shared memory: above {STATIC_SMEM_LIMIT}, emmy uses a dynamic pool")

    array = SharedArray(smem.name, f"s_{smem.name}", extents, dtype, nbytes, array_bytes)
    w.smem[smem.name] = array
    align = smem.align or nbytes
    w.emitter.declare_shared(f".shared .align {align} .b8 {array.symbol}[{array.size_bytes}]")
    w.emitter.comment(smem.pretty("")[0])


def emit_sync(w: Walker, sync: Sync) -> None:
    """A barrier: threads wait here until every participant arrives, and
    their earlier shared-memory writes are visible to the others after it.

    [EMMY] Sync docstring and render, three forms:
      - barrier_id == 0:      __syncthreads(), CTA-wide.
                              [DOC] TASK.md: bar.sync 0.
      - warp=True:            __syncwarp(), the warp only (full-warp mask, the
                              __syncwarp() default) -> bar.warp.sync 0xffffffff.
      - barrier_id in 1..15:  named barrier over exactly `count` threads,
                              rendered by emmy as bar.sync <id>, <count>.
    [OBS] every Sync in the open set is the CTA-wide one.
    """
    em = w.emitter
    em.comment(sync.pretty("")[0])
    if sync.warp:
        if sync.barrier_id != 0:
            raise ValueError("Sync(warp=True) cannot carry a named barrier_id")
        em.emit("bar.warp.sync 0xffffffff")
    elif sync.barrier_id == 0:
        em.emit("bar.sync 0")
    else:
        if sync.count is None or not 1 <= sync.barrier_id <= 15:
            raise ValueError(f"named Sync needs an id in 1..15 and a count, got "
                             f"id={sync.barrier_id} count={sync.count}")
        em.emit(f"bar.sync {sync.barrier_id}, {sync.count}")
