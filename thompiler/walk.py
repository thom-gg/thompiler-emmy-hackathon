"""Stage 3 - Walk: visit the kernel body statement by statement.

The Walker does two things:
  - dispatch: find the handler for a statement's class, or fail loudly
  - context:  carry the emitter and the kernel info to every handler

Block handlers (Tile, loops, Cond) recurse with walker.body(stmt.body), so
handler modules never import this one: no circular import.

A statement with no handler raises NotImplementedError naming it, so running
the compiler over every problem shows which statement each one still needs.
"""

from collections.abc import Callable, Iterable
from typing import Any

from emmy.compiler.ir.stmt import Stmt

from thompiler.emitter import Emitter
from thompiler.load import KernelInfo
from thompiler.handlers.memory import SharedArray
from thompiler.handlers.mma import Fragment


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


# Handler modules import Walker only for type hints, so importing them here,
# after Walker is defined, is not circular.
from emmy.compiler.ir.kernel.ir import (  # noqa: E402
    CpAsyncCommit, CpAsyncCopy, CpAsyncWait, LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore, Smem, Sync,
    Tile, TreeHalve, WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop  # noqa: E402
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write  # noqa: E402

from thompiler.handlers.blocks import emit_cond, emit_loop, emit_strided_loop, emit_tile  # noqa: E402
from thompiler.handlers.compute import emit_assign, emit_let, emit_load, emit_write  # noqa: E402
from thompiler.handlers.cp_async import (  # noqa: E402
    emit_cp_async_commit, emit_cp_async_copy, emit_cp_async_wait,
)
from thompiler.handlers.memory import emit_smem, emit_sync  # noqa: E402
from thompiler.handlers.mma import emit_ldmatrix_load, emit_mma_sync, emit_reg_fragment, emit_reg_store  # noqa: E402
from thompiler.handlers.reduce import emit_accum, emit_init, emit_tree_halve, emit_warp_shuffle  # noqa: E402

# Statement class -> handler. Keyed by the class itself (imported from emmy),
# not its name, so a typo is an import error rather than a silently unmatched key.
# A handler takes its own statement class (emit_load takes a Load), which a
# Callable[[Walker, Stmt], None] would reject: a function of Load cannot take
# any Stmt. The dispatch on type(stmt) is what guarantees the match, so the
# statement argument is typed Any here.
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
