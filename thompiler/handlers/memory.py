"""Handlers for shared memory and barriers (and, later, async copies)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.kernel.ir import Smem, Sync

from thompiler.types import SMEM_C_TYPES, PtxType

if TYPE_CHECKING:
    from thompiler.walk import Walker

# [DOC] PROVENANCE.md: shared arrays up to 48 KiB are static; above that emmy
# uses one dynamic pool sized by meta.json's smem_bytes ([OBS] only m01).
STATIC_SMEM_LIMIT = 48 * 1024

# The dynamic pool: one unsized .extern .shared array; each Smem array is a
# region at a fixed byte offset in it. [CUDA guide] every extern __shared__
# variable starts at the same address, so arrays cannot get their own.
# [EMMY] m01's .ref.cu aligns the pool to 16 bytes.
POOL_SYMBOL = "s_pool"
POOL_ALIGN = 16


@dataclass
class SharedArray:
    name: str                   # IR name, e.g. "acc0_smem"
    symbol: str                 # PTX symbol the array is addressed from
    offset: int                 # byte offset of the array from `symbol`
    extents: tuple[int, ...]    # used to flatten Load / Write indices
    dtype: PtxType              # element type (dtype.nbytes: size of one element)
    size_bytes: int             # size of the whole array: prod(extents) * dtype.nbytes


def emit_smem(w: Walker, smem: Smem) -> None:
    """Declare one shared array for the block, and record it so that Load /
    Write on its name use shared memory.

    [EMMY] Smem docstring: renders to `__shared__ [__align__(N)] <dtype>
    name[prod(extents)]`; extents flatten Load / Write indices; align = 0
    means the dtype's natural alignment.

    Two modes, chosen by the launch contract (meta.json's smem_bytes):
      - static (smem_bytes == 0): one sized .shared array per Smem, its own
        symbol "s_<name>", offset 0. Module-level, like
        examples/k_sum_1_reduce.ptx; the prefix keeps IR names off labels.
      - dynamic (smem_bytes > 0): every array is a region of the one pool,
        placed after the previous one at its alignment, like emmy's
        `_b_smem = _smem_pool + 24576` in m01. The pool must fit them all.
    """
    # Smem.dtype is a C type name ("float"), not an IR dtype.
    if smem.dtype not in SMEM_C_TYPES:
        raise NotImplementedError(f"Smem {smem.name} of type {smem.dtype!r}")
    dtype = SMEM_C_TYPES[smem.dtype]
    extents = tuple(int(e) for e in smem.extents)

    known = w.smem.get(smem.name)
    if known is not None:
        if known.extents != extents or known.dtype != dtype:
            raise ValueError(f"Smem {smem.name} declared twice with different shapes")
        return

    array_bytes = prod(extents) * dtype.nbytes
    align = smem.align or dtype.nbytes
    pool_bytes = w.kernel.dynamic_smem_bytes

    if pool_bytes == 0:
        # Check that the size of all already declared smem arrays + this new one
        # doesnt exceed the static limit
        declared_bytes = sum(a.size_bytes for a in w.smem.values())
        total_bytes = declared_bytes + array_bytes
        if total_bytes > STATIC_SMEM_LIMIT:
            raise ValueError(f"{total_bytes} bytes of static shared memory, above "
                             f"{STATIC_SMEM_LIMIT}, but meta.json's smem_bytes is 0")
        array = SharedArray(smem.name, f"s_{smem.name}", 0, extents, dtype, array_bytes)
        w.emitter.declare_shared(f".shared .align {align} .b8 {array.symbol}[{array_bytes}]")
    else:
        if align > POOL_ALIGN:
            raise NotImplementedError(f"Smem {smem.name} needs align {align} > pool's {POOL_ALIGN}")
        if not w.smem:      # first array: declare the pool once
            w.emitter.declare_shared(f".extern .shared .align {POOL_ALIGN} .b8 {POOL_SYMBOL}[]")
        end_of_previous = max((a.offset + a.size_bytes for a in w.smem.values()), default=0)
        offset = -(-end_of_previous // align) * align       # rounded up to a multiple of align
        if offset + array_bytes > pool_bytes:
            raise ValueError(f"Smem {smem.name} ends at byte {offset + array_bytes}, "
                             f"past the {pool_bytes}-byte pool of meta.json's smem_bytes")
        array = SharedArray(smem.name, POOL_SYMBOL, offset, extents, dtype, array_bytes)

    w.smem[smem.name] = array
    w.emitter.comment(smem.pretty("")[0] + (f"  (pool + {array.offset})" if pool_bytes else ""))


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
