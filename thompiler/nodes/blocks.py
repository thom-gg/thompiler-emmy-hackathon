"""Handlers for block statements: statements that contain a body.

Each handler emits its own control flow and recurses with walker.body(...).
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Stmt
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum

from thompiler import expr
from thompiler.emitter import f32_literal

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
    ctaid, ntid, tid = em.temp("b32"), em.temp("b32"), em.temp("b32")
    gid = em.define("_gid", "b32")
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
        reg = em.define(axis.name, "b32")
        if i == 0: # outermost axis
            if stride == 1: # if there is only one axis and its stride is 1 then we can just mov gid in it, let's avoid dividing by 1
                em.emit(f"mov.u32 {reg}, {gid}")
            else:
                em.emit(f"div.u32 {reg}, {gid}, {stride}")
        elif stride == 1: # same as previous comment we can avoid a division, it's only a modulo operation
            em.emit(f"rem.u32 {reg}, {gid}, {extents[i]}")
        else:
            quotient = em.temp("b32")
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
        dtype = s.dtype.name if s.dtype is not None else "f32"   # emmy defaults to f32 too
        if dtype != "f32":
            raise NotImplementedError(f"Accum {s.name} of dtype {dtype}")
        if s.op.identity is None:
            raise ValueError(f"Accum {s.name}: op {s.op.name} has no identity")
        em.emit(f"mov.f32 {em.define(s.name, 'f32')}, {f32_literal(s.op.identity)}")


def emit_strided_loop(w: Walker, loop: StridedLoop) -> None:
    """for (axis = start; axis < bound; axis += step) body

    [EMMY] StridedLoop docstring: bound is `end` when set, else the axis
    extent. Like emmy, a non-literal bound is evaluated once, before the loop.
    [EMMY] unroll only adds `#pragma unroll` in emmy's CUDA: a hint, so a
    rolled loop is equivalent. Ignored. [OBS] it is always False anyway.
    [EMMY] seed: initialize the body's accumulators before the loop.
    Loop arithmetic is signed, like the C `int` loop of emmy's rendering.
    """
    em = w.emitter
    em.comment(loop.pretty("")[0])

    if loop.seed:
        seed_accumulators(w, loop.body)

    # Evaluate start, bound and step once, before the loop.
    start = expr.int_expr(w, loop.start)
    if loop.end is not None:
        bound = expr.int_expr(w, loop.end)
    elif loop.axis.extent.is_static:
        bound = str(loop.axis.extent.value)
    else:
        raise NotImplementedError(f"StridedLoop over {loop.axis.name} with a non-static extent")
    step = expr.int_expr(w, loop.step) if isinstance(loop.step, Expr) else str(int(loop.step))

    var = em.define(loop.axis.name, "b32")
    head, done = em.label("loop"), em.label("loop_end")
    em.emit(f"mov.s32 {var}, {start}")
    em.place_label(head)
    stop = em.temp("pred")
    em.emit(f"setp.ge.s32 {stop}, {var}, {bound}")
    em.emit(f"@{stop} bra {done}")
    w.body(loop.body)
    em.emit(f"add.s32 {var}, {var}, {step}")
    em.emit(f"bra {head}")
    em.place_label(done)


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
