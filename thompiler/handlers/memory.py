"""Handlers for shared memory and barriers"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.kernel.ir import Smem, Sync

from thompiler.lowering.types import SMEM_C_TYPES, PtxType

if TYPE_CHECKING:
    from thompiler.walk import Walker

# PROVENANCE.md: static shared arrays can use up to 48 KiB of smem
STATIC_SMEM_LIMIT = 48 * 1024

# The dynamic pool: one unsized .extern .shared array; each Smem array is a
# region at a fixed byte offset in it.
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
    Write on its name know that name corresponds to shared memory.

    Two modes, chosen by the launch contract (meta.json's smem_bytes):
      - static (smem_bytes == 0): one sized .shared array per Smem, its own
        symbol "s_<name>", offset 0.
      - dynamic (smem_bytes > 0): every array is a region of the one pool,
        placed after the previous one at its alignment. The pool must fit them all.
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

    if pool_bytes == 0: # static mode (read from meta.json)
        # Check that the size of all already declared smem arrays + this new one doesnt exceed the static limit
        declared_bytes = sum(a.size_bytes for a in w.smem.values())
        total_bytes = declared_bytes + array_bytes
        if total_bytes > STATIC_SMEM_LIMIT:
            raise ValueError(f"{total_bytes} bytes of static shared memory, above "
                             f"{STATIC_SMEM_LIMIT}, but meta.json's smem_bytes is 0")
        array = SharedArray(smem.name, f"s_{smem.name}", 0, extents, dtype, array_bytes)
        w.emitter.declare_shared(f".shared .align {align} .b8 {array.symbol}[{array_bytes}]")
    else: # dynamic mode (read from meta.json)
        if align > POOL_ALIGN:
            raise NotImplementedError(f"Smem {smem.name} needs align {align} > pool's {POOL_ALIGN}")
        if not w.smem: # first time we walk on a Smem node: declare the pool once
            w.emitter.declare_shared(f".extern .shared .align {POOL_ALIGN} .b8 {POOL_SYMBOL}[]")
        end_of_previous = max((a.offset + a.size_bytes for a in w.smem.values()), default=0)
        offset = _round_up(end_of_previous, align)
        if offset + array_bytes > pool_bytes:
            raise ValueError(f"Smem {smem.name} ends at byte {offset + array_bytes}, "
                             f"past the {pool_bytes}-byte pool of meta.json's smem_bytes")
        array = SharedArray(smem.name, POOL_SYMBOL, offset, extents, dtype, array_bytes)

    w.smem[smem.name] = array
    w.emitter.comment(smem.pretty("")[0] + (f"  (pool + {array.offset})" if pool_bytes else ""))


def emit_sync(w: Walker, sync: Sync) -> None:
    """A barrier: threads wait here until every participant arrives, and
    their earlier shared-memory writes are visible to the others after it.

    three forms:
      - barrier_id == 0:      __syncthreads(), CTA-wide.
      - warp=True:            __syncwarp(), the warp only.
      - barrier_id in 1..15:  named barrier over exactly `count` threads,
                              rendered by emmy as bar.sync <id>, <count>.
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


def _round_up(n: int, multiple: int) -> int:
    """The smallest multiple of `multiple` that is >= n."""
    return (n + multiple - 1) // multiple * multiple
