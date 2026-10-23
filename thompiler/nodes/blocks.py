"""Handlers for block statements: statements that contain a body.

Each handler emits its own control flow and recurses with walker.body(...).
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from emmy.compiler.ir.kernel.ir import Tile

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
