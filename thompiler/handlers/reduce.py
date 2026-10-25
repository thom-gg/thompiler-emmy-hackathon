"""Handlers for reduction statements: accumulators and cross-thread combines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.leaves import Accum, Init

from thompiler.handlers.compute import emit_assign
from thompiler.handlers.memory import emit_sync
from thompiler.lowering import address, expr, ops
from thompiler.lowering.types import data_type

if TYPE_CHECKING:
    from thompiler.walk import Walker


def emit_accum(w: Walker, accum: Accum) -> None:
    """name <- op(name, value): fold one value into a running accumulator.

    Unlike an Assign, the accumulator is not a fresh value: it lives across
    the loop iterations, in one register that this statement overwrites.
    
    """
    em = w.emitter
    em.comment(accum.pretty("")[0])

    op = accum.op.name
    if op not in ops.ACCUM_OPS:
        raise NotImplementedError(f"Accum with op {op!r}")
    t = data_type(accum.dtype, f"Accum {accum.name}")
    if em.type_of(accum.value) != t:
        raise TypeError(f"Accum {accum.name}: {t.name} accumulator, "
                        f"{em.type_of(accum.value).name} value")

    # lookup, not define: the accumulator must already exist (seeded by its loop).
    acc = em.lookup(accum.name)
    left = em.lookup(accum.base) if accum.base is not None else acc
    em.emit(f"{ops.instruction(ops.BINARY, op, t)} {acc}, {left}, {em.lookup(accum.value)}")



def emit_init(w: Walker, init: Init) -> None:
    """Init(f32 name = identity): declare an accumulator and set it to the
    identity of its fold, here, once.
    """
    em = w.emitter
    em.comment(init.pretty("")[0])
    t = data_type(init.dtype, f"Init {init.name}")
    em.emit(f"mov.{t.mem} {em.define(init.name, t.reg)}, {t.literal(init.identity)}")


WARP_SIZE = 32


def emit_warp_shuffle(w: Walker, shuffle: WarpShuffle) -> None:
    """Combine each lane's partial state across `length` lanes of a warp,
    through registers only (no shared memory, no barrier).

    It's a reduction taking a XOR butterfly form. 
    """
    em = w.emitter
    em.comment(shuffle.pretty("")[0])

    length = int(shuffle.length)
    if length < 1 or length > WARP_SIZE or length & (length - 1):
        raise ValueError(f"WarpShuffle length {length} is not a power of two <= {WARP_SIZE}")
    t = data_type(shuffle.dtype, "WarpShuffle")
    if t.nbytes != 4:
        raise NotImplementedError(f"WarpShuffle of {t.name}: shfl moves 32 bits")

    mask = em.temp("b32")
    em.emit(f"activemask.b32 {mask}")
    step = length // 2
    while step > 0:
        em.comment(f"butterfly step: exchange with lane ^ {step}")
        for state, other in zip(shuffle.state, shuffle.state_b, strict=True):
            em.emit(f"shfl.sync.bfly.b32 {em.define(other, t.reg)}, {em.lookup(state)}, "
                    f"{step}, 0x1f, {mask}")
        for assign in shuffle.combine_states:
            emit_assign(w, assign)
        step //= 2


def emit_tree_halve(w: Walker, tree: TreeHalve) -> None:
    """Combine partial states across the warps of a block, through a tree in
    shared memory; afterwards every thread holds the full reduction.
    """
    em = w.emitter
    em.comment(tree.pretty("")[0])

    length = int(tree.length)
    if length < 1 or length & (length - 1):
        raise ValueError(f"TreeHalve length {length} is not a power of two")
    t = data_type(tree.dtype, "TreeHalve")
    barrier = Sync(barrier_id=tree.barrier_id, count=tree.barrier_count)

    def slot(index: Expr) -> tuple[Expr, ...]:
        """Shared-array index of tree slot `index`."""
        if tree.inner is None:
            return (index,)
        inner_var, scale = tree.inner
        return (BinaryExpr("+", BinaryExpr("*", index, Literal(scale, "int")), Var(inner_var)),)

    tid = Var(tree.tid_var)
    step = length // 2
    while step > 0:
        em.comment(f"tree step: {tree.tid_var} < {step} folds slot {tree.tid_var} + {step}")
        active = expr.pred_expr(w, BinaryExpr("<", tid, Literal(step, "int")))
        skip = em.label("tree_skip")
        em.emit(f"@!{active} bra {skip}")
        partner = BinaryExpr("+", tid, Literal(step, "int"))
        # 3 loops: read, compute / assign, store
        for buf, state, other in zip(tree.bufs, tree.state, tree.state_b, strict=True):
            space, own_addr, _ = address.memory_access(w, buf, slot(tid))
            em.emit(f"ld.{space}.{t.mem} {em.define(state, t.reg)}, [{own_addr}]")
            space, partner_addr, _ = address.memory_access(w, buf, slot(partner))
            em.emit(f"ld.{space}.{t.mem} {em.define(other, t.reg)}, [{partner_addr}]")
        for assign in tree.combine_states:
            emit_assign(w, assign)
        for buf, state in zip(tree.bufs, tree.state, strict=True):
            space, own_addr, _ = address.memory_access(w, buf, slot(tid))
            em.emit(f"st.{space}.{t.mem} [{own_addr}], {em.lookup(state)}")
        em.place_label(skip)
        emit_sync(w, barrier)       # outside the if: every thread of the block reaches it
        step //= 2

    root = Var(tree.inner[0]) if tree.inner is not None else Literal(0, "int")
    for buf, state in zip(tree.bufs, tree.state, strict=True):
        space, root_addr, _ = address.memory_access(w, buf, (root,))
        em.emit(f"ld.{space}.{t.mem} {em.lookup(state)}, [{root_addr}]")
