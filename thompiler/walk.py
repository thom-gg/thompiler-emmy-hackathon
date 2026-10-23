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

from emmy.compiler.ir.stmt import Stmt

from thompiler.emitter import Emitter
from thompiler.load import KernelInfo


class Walker:
    def __init__(self, emitter: Emitter, kernel: KernelInfo) -> None:
        self.emitter = emitter
        self.kernel = kernel

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
from emmy.compiler.ir.kernel.ir import Tile  # noqa: E402
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write  # noqa: E402

from thompiler.nodes.blocks import emit_tile  # noqa: E402
from thompiler.nodes.compute import emit_assign, emit_load, emit_write  # noqa: E402

# Statement class -> handler. Keyed by the class itself (imported from emmy),
# not its name, so a typo is an import error rather than a silently unmatched key.
Handler = Callable[[Walker, Stmt], None]
HANDLERS: dict[type[Stmt], Handler] = {
    Tile: emit_tile,
    Load: emit_load,
    Assign: emit_assign,
    Write: emit_write,
}
