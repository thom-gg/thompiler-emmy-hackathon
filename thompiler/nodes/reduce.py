"""Handlers for reduction statements: accumulators and cross-thread combines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.leaves import Accum

from thompiler import expr
from thompiler.nodes.compute import emit_assign, memory_access
from thompiler.nodes.memory import emit_sync

if TYPE_CHECKING:
    from thompiler.walk import Walker

# Fold op -> PTX f32 instruction. IEEE .rn like the elementwise ops (see compute.py).
# [EMMY] Accum docstring: op is "typically one of ADD / MAX / MIN / MUL".
# [OBS] only add and maximum are used.
ACCUM_F32: dict[str, str] = {
    "add": "add.rn.f32",        # acc += v
    "multiply": "mul.rn.f32",   # acc *= v
    "maximum": "max.f32",       # acc = fmaxf(acc, v)
    "minimum": "min.f32",       # acc = fminf(acc, v)
}


def emit_accum(w: Walker, accum: Accum) -> None:
    """name <- op(name, value): fold one value into a running accumulator.

    Unlike an Assign, the accumulator is not a fresh value: it lives across
    the loop iterations, in one register that this statement overwrites.
    [EMMY] Accum docstring: it starts at op.identity before the first
    iteration (the enclosing loop seeds it, see blocks.seed_accumulators)
    and holds the reduced value after the loop.
    [EMMY] `base`, when set, replaces the left operand: name = op(base, value).
    It is a rescaled copy of the state bound by a preceding Assign (online
    softmax, f02 / a01 / a02); the seed is unchanged.
    """
    em = w.emitter
    em.comment(accum.pretty("")[0])

    op = accum.op.name
    if op not in ACCUM_F32:
        raise NotImplementedError(f"Accum with op {op!r}")
    dtype = accum.dtype.name if accum.dtype is not None else "f32"   # emmy defaults to f32
    if dtype != "f32" or em.kind_of(accum.value) != "f32":
        raise NotImplementedError(f"Accum {accum.name} of dtype {dtype}")

    # lookup, not define: the accumulator must already exist (seeded by its loop).
    acc = em.lookup(accum.name)
    left = em.lookup(accum.base) if accum.base is not None else acc
    em.emit(f"{ACCUM_F32[op]} {acc}, {left}, {em.lookup(accum.value)}")


WARP_SIZE = 32


def emit_warp_shuffle(w: Walker, shuffle: WarpShuffle) -> None:
    """Combine each lane's partial state across `length` lanes of a warp,
    through registers only (no shared memory, no barrier).

    [EMMY] WarpShuffle docstring and render: an XOR butterfly. For
    s = length/2, length/4, ..., 1, every lane reads each state component of
    lane (lane XOR s) into state_b, then runs combine_states (a list of
    Assigns) that merges state_b into state, in place. After the last step
    every lane holds the full reduction, under the same names.
    [EMMY] length is a power of two <= warp size; the butterfly never leaves
    an aligned group of `length` lanes, so a shorter length is a per-segment
    reduce with the same code.
    [EMMY] the lane mask is the active mask (__activemask()), not 0xffffffff:
    naming absent lanes of a partial warp is undefined behavior.
    [PTX] shfl.sync.bfly.b32 d, a, b, c, membermask: d = a of lane (lane ^ b);
    c = 0x1f is the clamp for a whole-warp butterfly. A .b32 instruction
    accepts .f32 registers (bit-size types match any type of the same size).
    """
    em = w.emitter
    em.comment(shuffle.pretty("")[0])

    length = int(shuffle.length)
    if length < 1 or length > WARP_SIZE or length & (length - 1):
        raise ValueError(f"WarpShuffle length {length} is not a power of two <= {WARP_SIZE}")
    if shuffle.dtype.name != "f32":
        raise NotImplementedError(f"WarpShuffle of dtype {shuffle.dtype.name}")

    mask = em.temp("b32")
    em.emit(f"activemask.b32 {mask}")
    step = length // 2
    while step > 0:
        em.comment(f"butterfly step: exchange with lane ^ {step}")
        for state, other in zip(shuffle.state, shuffle.state_b, strict=True):
            em.emit(f"shfl.sync.bfly.b32 {em.define(other, 'f32')}, {em.lookup(state)}, "
                    f"{step}, 0x1f, {mask}")
        for assign in shuffle.combine_states:
            emit_assign(w, assign)
        step //= 2


def emit_tree_halve(w: Walker, tree: TreeHalve) -> None:
    """Combine partial states across the warps of a block, through a tree in
    shared memory; afterwards every thread holds the full reduction.

    [EMMY] TreeHalve docstring and render: each state component has a shared
    slab bufs[i][0..length), already filled by earlier statements (r01: one
    partial per warp, acc0_smem[warp] = acc0). For s = length/2, ..., 1:
        if (t < s) { state = buf[t]; state_b = buf[t + s];
                     combine_states; buf[t] = state; }
        barrier
    then every thread reads the result back: state = buf[0].
    t is the variable named by tid_var ([OBS] "warp"). length is a constant,
    so the steps are unrolled here; emmy writes the same steps as a C loop.
    [EMMY] the barrier is __syncthreads(), or bar.sync <id>, <count> when
    barrier_id > 0. inner = (var, scale) makes slot t be t * scale + var.
    The tree overwrites the state registers; the final read restores them.
    """
    em = w.emitter
    em.comment(tree.pretty("")[0])

    length = int(tree.length)
    if length < 1 or length & (length - 1):
        raise ValueError(f"TreeHalve length {length} is not a power of two")
    if tree.dtype.name != "f32":
        raise NotImplementedError(f"TreeHalve of dtype {tree.dtype.name}")
    barrier = Sync(barrier_id=tree.barrier_id, count=tree.barrier_count)

    def slot(index: object) -> tuple:
        """Shared-array index of tree slot `index` (an Expr)."""
        if tree.inner is None:
            return (index,)
        inner_var, scale = tree.inner
        return (BinaryExpr("+", BinaryExpr("*", index, Literal(scale, "int")), Var(inner_var)),)

    t = Var(tree.tid_var)
    step = length // 2
    while step > 0:
        em.comment(f"tree step: {tree.tid_var} < {step} folds slot {tree.tid_var} + {step}")
        active = expr.pred_expr(w, BinaryExpr("<", t, Literal(step, "int")))
        skip = em.label("tree_skip")
        em.emit(f"@!{active} bra {skip}")
        partner = BinaryExpr("+", t, Literal(step, "int"))
        for buf, state, other in zip(tree.bufs, tree.state, tree.state_b, strict=True):
            space, own_addr = memory_access(w, buf, slot(t))
            em.emit(f"ld.{space}.f32 {em.define(state, 'f32')}, [{own_addr}]")
            space, partner_addr = memory_access(w, buf, slot(partner))
            em.emit(f"ld.{space}.f32 {em.define(other, 'f32')}, [{partner_addr}]")
        for assign in tree.combine_states:
            emit_assign(w, assign)
        for buf, state in zip(tree.bufs, tree.state, strict=True):
            space, own_addr = memory_access(w, buf, slot(t))
            em.emit(f"st.{space}.f32 [{own_addr}], {em.lookup(state)}")
        em.place_label(skip)
        emit_sync(w, barrier)       # outside the if: every thread of the block reaches it
        step //= 2

    root = Var(tree.inner[0]) if tree.inner is not None else Literal(0, "int")
    for buf, state in zip(tree.bufs, tree.state, strict=True):
        space, root_addr = memory_access(w, buf, (root,))
        em.emit(f"ld.{space}.f32 {em.lookup(state)}, [{root_addr}]")
