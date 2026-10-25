"""Stage 3 - Walk: visit the kernel body statement by statement.

The Walker does two things:
  - dispatch: find the handler for a statement's class, or fail loudly
  - context:  carry the emitter and the kernel info to every handler
"""

from collections.abc import Callable, Iterable
from typing import Any

from emmy.compiler.ir.stmt import Stmt
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit, CpAsyncCopy, CpAsyncWait, LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore, Smem, Sync,
    Tile, TreeHalve, WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write

from thompiler.emitter import Emitter
from thompiler.load import KernelInfo
from thompiler.handlers.blocks import emit_cond, emit_loop, emit_strided_loop, emit_tile
from thompiler.handlers.compute import emit_assign, emit_let, emit_load, emit_write
from thompiler.handlers.cp_async import (
    emit_cp_async_commit, emit_cp_async_copy, emit_cp_async_wait,
)
from thompiler.handlers.memory import SharedArray, emit_smem, emit_sync
from thompiler.handlers.mma import Fragment, emit_ldmatrix_load, emit_mma_sync, emit_reg_fragment, emit_reg_store
from thompiler.handlers.reduce import emit_accum, emit_init, emit_tree_halve, emit_warp_shuffle


class Walker:
    def __init__(self, emitter: Emitter, kernel: KernelInfo) -> None:
        self.emitter = emitter
        self.kernel = kernel
        # Shared arrays declared so far by Smem, by IR name: Load / Write read
        # it to tell a shared-memory buffer from a kernel param.
        self.smem: dict[str, SharedArray] = {}
        # Tensor-core fragments declared so far by RegFragment, by IR name.
        self.fragments: dict[str, Fragment] = {}
        # Variables of the loops being unrolled, with their value in the
        # current copy of the body: int_expr reads them as constants.
        self.consts: dict[str, int] = {}

    def body(self, stmts: Iterable[Stmt]) -> None:
        for stmt in stmts:
            self.stmt(stmt)

    def stmt(self, stmt: Stmt) -> None:
        handler = HANDLERS.get(type(stmt))
        if handler is None:
            raise NotImplementedError(f"IR statement {type(stmt).__name__} has no handler yet")
        handler(self, stmt)


# Statement class -> handler. Keyed by the class itself (imported from emmy),
# A handler takes its own statement class (emit_load takes a Load)
Handler = Callable[[Walker, Any], None]
HANDLERS: dict[type[Stmt], Handler] = {
    Tile: emit_tile,
    StridedLoop: emit_strided_loop,
    Loop: emit_loop,
    Cond: emit_cond,
    Load: emit_load,
    Assign: emit_assign,
    Let: emit_let,
    Write: emit_write,
    Accum: emit_accum,
    Init: emit_init,
    WarpShuffle: emit_warp_shuffle,
    TreeHalve: emit_tree_halve,
    Smem: emit_smem,
    Sync: emit_sync,
    CpAsyncCopy: emit_cp_async_copy,
    CpAsyncCommit: emit_cp_async_commit,
    CpAsyncWait: emit_cp_async_wait,
    RegFragment: emit_reg_fragment,
    LdmatrixLoad: emit_ldmatrix_load,
    MmaSyncPtx: emit_mma_sync,
    RegStore: emit_reg_store,
}
