"""Handlers for asynchronous copies from global to shared memory (cp.async)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait

from thompiler.lowering import address

if TYPE_CHECKING:
    from thompiler.walk import Walker

# [EMMY] CpAsyncCopy docstring: the cp.async copy sizes; 16-byte copies use
# .cg (cache-global, bypass L1, 16 bytes only), 4 / 8-byte copies use .ca.
CP_ASYNC_CACHE: dict[int, str] = {4: "ca", 8: "ca", 16: "cg"}


def emit_cp_async_copy(w: Walker, copy: CpAsyncCopy) -> None:
    """cp.async[N B] smem[smem_index] (+ lane) <- src[src_index]: copy N
    contiguous bytes from global to shared memory, without going through a
    register; the copy completes in the background.    
    """
    em = w.emitter
    em.comment(copy.pretty("")[0])

    if copy.nbytes not in CP_ASYNC_CACHE:
        raise ValueError(f"cp.async of {copy.nbytes} bytes (4, 8 or 16 only)")
    array = w.smem.get(copy.smem)
    source = address.param(w, copy.src)
    if array is None or source is None:
        raise ValueError(f"cp.async from {copy.src!r} to {copy.smem!r}: "
                         "needs a kernel param source and an Smem destination")

    # lane_index is each individual thread's offset
    smem_index = tuple(copy.smem_index)
    if copy.lane_index is not None:
        lane_row, lane_col = copy.lane_index
        *outer, row, col = smem_index
        smem_index = (*outer, BinaryExpr("+", row, lane_row), BinaryExpr("+", col, lane_col))

    dst = address.shared_address(w, array, smem_index, copy.swizzle)
    src = address.global_address(w, source, copy.src_index)
    em.emit(f"cp.async.{CP_ASYNC_CACHE[copy.nbytes]}.shared.global [{dst}], [{src}], {copy.nbytes}")


def emit_cp_async_commit(w: Walker, commit: CpAsyncCommit) -> None:
    """close this thread's preceding cp.async copies into one group."""
    w.emitter.emit("cp.async.commit_group")


def emit_cp_async_wait(w: Walker, wait: CpAsyncWait) -> None:
    """block this thread until at most `group` cp.async groups are
    still in flight (0 = all done)."""
    w.emitter.emit(f"cp.async.wait_group {int(wait.group)}")
