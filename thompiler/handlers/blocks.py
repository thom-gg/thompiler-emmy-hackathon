"""Handlers for block statements: statements that contain a body.

Each handler emits its own control flow and recurses with walker.body(...).
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum

from thompiler.handlers.mma import indexes_fragment
from thompiler.lowering import expr
from thompiler.lowering.types import I32, data_type

if TYPE_CHECKING:
    from thompiler.walk import Walker


def emit_tile(w: Walker, tile: Tile) -> None:
    """One thread per cell of the iteration space (the product of the axes'
    extents). Each thread computes its global id, skips the body if it is
    past the last cell, decodes its axis indices, and runs the body.

    [EMMY] Tile docstring and its CUDA rendering (emmy/compiler/ir/kernel/ir.py):
      _gid = blockIdx.x * blockDim.x + threadIdx.x
      axis_i = (_gid / stride_i) % extent_i, stride_i = product of the extents
               after axis i; the outermost axis needs no modulo.
    [DOC] the guard _gid < N is not in the IR (TASK.md); we always emit it.
          Emmy omits it when it is always true; emitting it then is harmless.
    [EMMY] raster_group / raster_axes only reorder which CTA computes which
          block tile, for L2 reuse: a permutation, so the plain decode gives
          the same results. Ignored here (performance only).
    [EMMY] aux_threads changes the _gid decode. [OBS] it is 0 everywhere,
          so a nonzero value fails loudly.
    """
    em = w.emitter
    if tile.aux_threads != 0:
        raise NotImplementedError(f"Tile with aux_threads={tile.aux_threads}")
    # [EMMY] Tile.extents: each axis' static extent (raises NotImplementedError
    # on a symbolic one). Tile.n_elements: their product - the "N=" that the
    # .ir.txt pretty-printer shows (Tile.pretty prints n_elements).
    extents = tile.extents
    n_cells = tile.n_elements
    if n_cells > 2**31 - 1:     # Emmy switches _gid to 64-bit past this
        raise NotImplementedError(f"Tile with N={n_cells} needs a 64-bit thread id")

    em.comment(f"Tile[{', '.join(a.name for a in tile.axes)}] (N={n_cells})")

    # _gid = %ctaid.x * %ntid.x + %tid.x
    # The decode is unsigned (u32): _gid and every axis index are >= 0.
    ctaid, ntid, tid = em.temp(I32.reg), em.temp(I32.reg), em.temp(I32.reg)
    gid = em.define("_gid", I32.reg)
    em.emit(f"mov.u32 {ctaid}, %ctaid.x")
    em.emit(f"mov.u32 {ntid}, %ntid.x")
    em.emit(f"mov.u32 {tid}, %tid.x")
    em.emit(f"mad.lo.u32 {gid}, {ctaid}, {ntid}, {tid}")

    # Guard: threads past the last cell jump over the whole body.
    past_end = em.temp("pred")
    end = em.label("tile_end")
    em.emit(f"setp.ge.u32 {past_end}, {gid}, {n_cells}")
    em.emit(f"@{past_end} bra {end}")

    # Decode each axis index from _gid, outermost first.
    em.comment("START: decoding each axis index")
    for i, axis in enumerate(tile.axes):
        stride = prod(extents[i + 1:])
        reg = em.define(axis.name, I32.reg)
        if i == 0: # outermost axis
            if stride == 1: # if there is only one axis and its stride is 1 then we can just mov gid in it, let's avoid dividing by 1
                em.emit(f"mov.u32 {reg}, {gid}")
            else:
                em.emit(f"div.u32 {reg}, {gid}, {stride}")
        elif stride == 1: # same as previous comment we can avoid a division, it's only a modulo operation
            em.emit(f"rem.u32 {reg}, {gid}, {extents[i]}")
        else:
            quotient = em.temp(I32.reg)
            em.emit(f"div.u32 {quotient}, {gid}, {stride}")
            em.emit(f"rem.u32 {reg}, {quotient}, {extents[i]}")

    em.comment("END: decoding each axis index")

    w.body(tile.body)
    em.place_label(end)


def seed_accumulators(w: Walker, body: tuple[Stmt, ...]) -> None:
    """Before a loop: set each accumulator its body folds to the identity of
    its op (0 for add, 1 for multiply, -1e30 for maximum).

    [EMMY] Accum docstring: "Before the first iteration name is initialized
    to op.identity". [EMMY] Loop / StridedLoop.render emit that seed for
    each distinct Accum of the loop's *immediate* body, unless seed=False.
    """
    em = w.emitter
    seen: set[str] = set()
    for s in body:
        if not isinstance(s, Accum) or s.name in seen:
            continue
        seen.add(s.name)
        t = data_type(s.dtype, f"Accum {s.name}")
        if s.op.identity is None:
            raise ValueError(f"Accum {s.name}: op {s.op.name} has no identity")
        em.emit(f"mov.{t.mem} {em.define(s.name, t.reg)}, {t.literal(s.op.identity)}")


def emit_strided_loop(w: Walker, loop: StridedLoop) -> None:
    """for (axis = start; axis < bound; axis += step) body

    [EMMY] StridedLoop docstring: bound is `end` when set, else the axis
    extent. Like emmy, a non-literal bound is evaluated once, before the loop.
    [EMMY] seed: initialize the body's accumulators before the loop.
    """
    em = w.emitter
    em.comment(loop.pretty("")[0])

    if loop.seed:
        seed_accumulators(w, loop.body)

    # Evaluate start, bound and step once, before the loop.
    start = expr.int_expr(w, loop.start)
    if loop.end is not None:
        bound = expr.int_expr(w, loop.end)
    else:
        bound = _static_extent(loop.axis)
    step = expr.int_expr(w, loop.step) if isinstance(loop.step, Expr) else str(int(loop.step))
    emit_counted_loop(w, loop.axis.name, start, bound, step, loop.body)


def emit_counted_loop(w: Walker, var_name: str, start: str, bound: str, step: str,
                      body: tuple[Stmt, ...]) -> None:
    """for (var = start; var < bound; var += step) body, with start / bound /
    step already evaluated to PTX operands: a PTX loop, or the body repeated
    once per value when the variable indexes a fragment.

    [EMMY] Loop / StridedLoop `unroll` only adds `#pragma unroll` in emmy's
    CUDA ("no effect on the IR's iteration semantics"): a hint, so a rolled
    loop is equivalent; ptxas may still unroll short loops. The exception is
    a loop whose variable indexes a fragment family ("_a0_s[_r0]"): [PTX]
    registers cannot be indexed at run time, so that loop must be unrolled
    here ([EMMY] nvcc unrolls it in emmy's CUDA, where the fragment is a C
    array). [OBS] those loops are 2 to 4 iterations, with unroll=True.
    """
    if indexes_fragment(var_name, body):
        emit_unrolled_loop(w, var_name, start, bound, step, body)
    else:
        emit_rolled_loop(w, var_name, start, bound, step, body)


def emit_unrolled_loop(w: Walker, var_name: str, start: str, bound: str, step: str,
                       body: tuple[Stmt, ...]) -> None:
    """The body once per value of the loop variable, which int_expr and the
    fragment references read as a constant (w.consts)."""
    em = w.emitter
    bounds = (start, bound, step)
    if any(b.startswith("%") for b in bounds):
        raise NotImplementedError(f"loop over {var_name} indexes a fragment but its bounds are not constants")
    first, end, stride = (int(b) for b in bounds)
    if stride <= 0:
        raise ValueError(f"loop over {var_name} with step {stride}")
    outer = w.consts.get(var_name)
    for value in range(first, end, stride):
        em.comment(f"unrolled: {var_name} = {value}")
        w.consts[var_name] = value
        w.body(body)
    if outer is None:
        del w.consts[var_name]
    else:
        w.consts[var_name] = outer


def emit_loop(w: Walker, loop: Loop) -> None:
    """for (axis = 0; axis < extent; axis++) body: one thread runs every value.

    [EMMY] Loop docstring and render: `for (int axis = 0; axis < extent;
    axis++)`, with the same per-Accum seed as StridedLoop (unless seed=False).
    [EMMY] `carries` is the carried state of Carry statements (*-marked in
    TASK.md, in no problem): a loop that carries any fails loudly.
    """
    em = w.emitter
    em.comment(loop.pretty("")[0])
    if loop.carries:
        raise NotImplementedError(f"Loop over {loop.axis.name} carrying {list(loop.carries)}")

    if loop.seed:
        seed_accumulators(w, loop.body)
    emit_counted_loop(w, loop.axis.name, "0", _static_extent(loop.axis), "1", loop.body)


def emit_rolled_loop(w: Walker, var_name: str, start: str, bound: str, step: str,
                     body: tuple[Stmt, ...]) -> None:
    """A PTX loop. The condition is tested before every iteration, like a C
    for loop. Signed, like emmy's C `int` loops."""
    em = w.emitter
    var = em.define(var_name, I32.reg)
    head, done = em.label("loop"), em.label("loop_end")
    em.emit(f"mov.{I32.ptx} {var}, {start}")
    em.place_label(head)
    stop = em.temp("pred")
    em.emit(f"setp.ge.{I32.ptx} {stop}, {var}, {bound}")
    em.emit(f"@{stop} bra {done}")
    w.body(body)
    em.emit(f"add.{I32.ptx} {var}, {var}, {step}")
    em.emit(f"bra {head}")
    em.place_label(done)


def _static_extent(axis) -> str:
    if not axis.extent.is_static:
        raise NotImplementedError(f"loop over {axis.name} with a non-static extent")
    return str(axis.extent.value)


def emit_cond(w: Walker, cond: Cond) -> None:
    """if (cond) { body } [else { else_body }]

    [EMMY] Cond docstring: cond is an Expr over axis vars and prior SSA
    names; an empty else_body is a bare if.
    [OBS] every condition is an == (lane == 0, a0_co == 0); none has an else.
    Threads for which cond is false jump over the body.
    """
    em = w.emitter
    em.comment(f"if {cond.cond.pretty()}")
    taken = expr.pred_expr(w, cond.cond)
    if cond.else_body:
        otherwise, end = em.label("else"), em.label("if_end")
        em.emit(f"@!{taken} bra {otherwise}")
        w.body(cond.body)
        em.emit(f"bra {end}")
        em.place_label(otherwise)
        w.body(cond.else_body)
        em.place_label(end)
    else:
        end = em.label("if_end")
        em.emit(f"@!{taken} bra {end}")
        w.body(cond.body)
        em.place_label(end)
