"""Handlers for tensor-core statements (mma.sync): fragments and their users.

A fragment is the part of an mma operand (A, B, or the accumulator C) that
one thread of the warp holds, in a few registers. The 32 threads' fragments
together form the tile; mma.sync combines them in one warp-wide instruction.

Fragment registers. [PTX] registers cannot be indexed by a runtime value,
while the IR addresses a fragment family by index ("_a0_s[_r0]"): every
register of every member gets its own name, %x_<fragment>_<member>_<slot>,
and the loops whose variable indexes a fragment are unrolled
(blocks.indexes_fragment), so the index is a constant by the time a
fragment is referenced.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore

from thompiler.lowering import address, expr
from thompiler.lowering.types import F16, F32, PtxType, RegKind, data_type

if TYPE_CHECKING:
    from thompiler.walk import Walker

# A fragment reference: "_c0_0", or a family member "_a0_s[0]" / "_a0_s[_r0]".
FRAGMENT_REF = re.compile(r"^(?P<name>\w+)(?:\[(?P<index>[^\]]+)\])?$")


@dataclass
class Fragment:
    name: str           # IR name of the fragment (family), e.g. "_a0_s"
    role: str           # "a", "b" or "c"
    dtype: PtxType      # element type: f16 for a / b, f32 (or f16) for c
    count: int          # members in the family (1: a single fragment)
    nregs: int          # registers per member
    reg: RegKind        # register kind: f32 for an f32 c, else b32 (2 halves packed)

    def registers(self, member: int) -> list[str]:
        return [f"%x_{self.name}_{member}_{slot}" for slot in range(self.nregs)]


def mma_sync_nregs(role: str, shape: tuple[int, int, int], dtype: PtxType) -> int:
    """Registers per lane of an mma.sync operand.

    [EMMY] _mma_sync_nregs (emmy/compiler/ir/kernel/ir.py): a and b pack two
    16-bit elements per 32-bit register, c holds one f32 per register or two
    packed halves. For m16n8k16 f16: a[4] (16x16 / 32 lanes / 2), b[2]
    (16x8 / 32 / 2), c[4] in f32 (16x8 / 32), c[2] in f16.
    """
    m, n, k = shape
    if role in ("a", "b"):
        if dtype.nbytes != 2:
            raise NotImplementedError(f"mma operand {role} of {dtype.name}")
        return (m * k if role == "a" else k * n) // (32 * 2)
    if role == "c":
        return (m * n) // (64 if dtype.nbytes == 2 else 32)
    raise ValueError(f"mma.sync: unsupported role {role!r}")


def emit_reg_fragment(w: Walker, frag: RegFragment) -> None:
    """Declare a fragment family: count members of nregs registers each.

    [EMMY] RegFragment docstring and render: a / b are `unsigned name[n]`
    (two halves per 32-bit register); c is `float name[n]` (or packed
    `unsigned` for an f16 accumulator) and "is zero-initialised at
    declaration, so the mma.sync path needs no separate fill node".
    `nregs`, when set, overrides the shape-derived count.
    [OBS] t01 / t02: m16n8k16, a / b f16, c f32; count 1, 2 or 4.
    """
    em = w.emitter
    em.comment(frag.pretty("")[0])
    if frag.name in w.fragments:
        raise ValueError(f"fragment {frag.name} declared twice")

    dtype = data_type(frag.dtype, f"RegFragment {frag.name}")
    m, n, k = frag.shape
    nregs = frag.nregs if frag.nregs else mma_sync_nregs(frag.role, (m, n, k), dtype)
    reg: RegKind = "f32" if frag.role == "c" and dtype == F32 else "b32"
    fragment = Fragment(frag.name, frag.role, dtype, int(frag.count), nregs, reg)
    w.fragments[frag.name] = fragment

    for member in range(fragment.count):
        for r in fragment.registers(member):
            em.declare(r, reg)
            if frag.role == "c":
                # f32 zero, or two packed f16 zeros (all bits 0 either way).
                em.emit(f"mov.{'f32' if reg == 'f32' else 'b32'} {r}, {'0f00000000' if reg == 'f32' else '0'}")


def fragment_registers(w: Walker, ref: str) -> tuple[Fragment, list[str]]:
    """The fragment and the registers of a fragment reference such as
    "_c0_0", "_a0_s[0]" or "_a0_s[_r0]" (with _r0 an unrolled loop variable)."""
    match = FRAGMENT_REF.match(ref)
    if match is None:
        raise ValueError(f"not a fragment reference: {ref!r}")
    fragment = w.fragments.get(match["name"])
    if fragment is None:
        raise KeyError(f"fragment {match['name']!r} is used before its RegFragment")
    member = 0 if match["index"] is None else _constant_index(w, match["index"], ref)
    if not 0 <= member < fragment.count:
        raise IndexError(f"{ref}: member {member} of a family of {fragment.count}")
    return fragment, fragment.registers(member)


def indexes_fragment(var: str, node: object) -> bool:
    """True when `var` appears in the index of a fragment reference
    somewhere under `node` ("_a0_s[_r0]" for var "_r0"): its loop must be
    unrolled for that index to be a constant."""
    if isinstance(node, str):
        match = FRAGMENT_REF.match(node)
        return bool(match and match["index"] and re.search(rf"\b{re.escape(var)}\b", match["index"]))
    if isinstance(node, (tuple, list)):
        return any(indexes_fragment(var, item) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return any(indexes_fragment(var, getattr(node, f.name)) for f in dataclasses.fields(node))
    return False


def _constant_index(w: Walker, text: str, ref: str) -> int:
    """Evaluate a fragment index: integer literals, unrolled loop variables
    (w.consts) and + - * between them. Anything else cannot be a register."""
    def value(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in w.consts:
            return w.consts[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = value(node.left), value(node.right)
            return left + right if isinstance(node.op, ast.Add) else (
                left - right if isinstance(node.op, ast.Sub) else left * right)
        raise NotImplementedError(f"{ref}: fragment index {text!r} is not a compile-time constant")
    return value(ast.parse(text, mode="eval").body)


def emit_ldmatrix_load(w: Walker, load: LdmatrixLoad) -> None:
    """Load mma operand fragments from a shared slab, cooperatively: every lane
    gives the address of one 16-byte row (8 halves) of an 8x8 matrix, and gets
    back, per matrix, the 2 halves of the fragment layout it holds.

    [PTX] ldmatrix.sync.aligned.m8n8.x{2,4}{.trans}.shared.b16: .x4 loads 4
    matrices (lanes 0-7 address matrix 0, 8-15 matrix 1, ...), .trans loads
    them column-major.
    [EMMY] LdmatrixLoad.render, the staged m16n8k16 forms. The lane's element
    is the tile base `flat(src_index)` plus its own offset, through the slab's
    swizzle (the same one its fill applied):
      a (16x16):            x4,       (lane % 16) * ldm + (lane / 16) * 8
      b pair, row-major:    x4.trans, (lane % 16) * ldm + (lane / 16) * 8
      b pair, b_trans:      x4,       ((lane % 8) + (lane / 16) * 8) * ldm + ((lane / 8) % 2) * 8
      b single, b_trans:    x2,       (lane % 8) * ldm + ((lane / 8) % 2) * 8
      b single, row-major:  x2.trans, (lane % 16) * ldm
    A pair loads two B fragments in one x4: registers 0-1 to frag, 2-3 to
    pair_frag. emmy splits the lane part out of the swizzle when it can
    (swizzled_slab_index), an optimization giving the same element.
    [EMMY] fragment_index only matters to the Volta m8n8k4 layout.
    [OBS] t01 / t02: staged, a in x4 and b in x4.trans pairs.
    """
    em = w.emitter
    em.comment(load.pretty("")[0])
    if (not load.staged or load.fragment_layout != "m16n8k16" or load.byte_slab
            or load.gmem_guard is not None or load.k_zero is not None or load.scale_buffer is not None):
        raise NotImplementedError(f"LdmatrixLoad form: staged={load.staged}, layout={load.fragment_layout}, "
                                  f"byte_slab={load.byte_slab}, guards or scales")
    array = w.smem.get(load.src_buffer)
    if array is None:
        raise ValueError(f"LdmatrixLoad from {load.src_buffer!r}, which is not an Smem array")
    if array.dtype.nbytes != 2:
        raise NotImplementedError(f"ldmatrix of {array.dtype.name}: it loads 16-bit elements")

    fragment, regs = fragment_registers(w, load.frag)
    if fragment.role != load.role:
        raise ValueError(f"LdmatrixLoad role {load.role} into a {fragment.role} fragment")
    if load.pair_frag is not None:
        if load.role != "b":
            raise ValueError("a paired ldmatrix loads two B fragments")
        regs = regs + fragment_registers(w, load.pair_frag)[1]

    ldm = load.ldm or array.extents[-1]
    lane = Var("lane")
    def lane_offset(row: Expr, col: Expr | None) -> Expr:
        offset = BinaryExpr("*", row, Literal(ldm, "int"))
        return offset if col is None else BinaryExpr("+", offset, col)
    def rem(e: Expr, n: int) -> Expr:
        return BinaryExpr("%", e, Literal(n, "int"))
    def div(e: Expr, n: int) -> Expr:
        return BinaryExpr("/", e, Literal(n, "int"))
    def times8(e: Expr) -> Expr:
        return BinaryExpr("*", e, Literal(8, "int"))

    if load.role == "a" or not load.b_trans:
        lane_row = rem(lane, 16)
    else:
        lane_row = rem(lane, 8) if load.pair_frag is None else BinaryExpr("+", rem(lane, 8), times8(div(lane, 16)))
    if load.role == "a" or (load.pair_frag is not None and not load.b_trans):
        lane_col: Expr | None = times8(div(lane, 16))
    elif load.b_trans:
        lane_col = times8(rem(div(lane, 8), 2))
    else:
        lane_col = None
    matrices = 4 if load.role == "a" or load.pair_frag is not None else 2
    trans = load.role == "b" and not load.b_trans
    if len(regs) != matrices:
        raise ValueError(f"ldmatrix .x{matrices} into {len(regs)} registers ({load.frag})")

    tile = address.flat_index(w, array.name, array.extents, tuple(load.src_index))
    element = expr.add_ints(w, tile, expr.int_expr(w, lane_offset(lane_row, lane_col)))
    addr = address.shared_element_address(w, array, element, load.swizzle)
    em.emit(f"ldmatrix.sync.aligned.m8n8.x{matrices}{'.trans' if trans else ''}.shared.b16 "
            f"{{{', '.join(regs)}}}, [{addr}]")


def emit_mma_sync(w: Walker, mma: MmaSyncPtx) -> None:
    """One warp-wide tensor-core multiply-accumulate: c = a @ b + c, the
    accumulator being both the input and the output.

    [PTX] mma.sync.aligned.m16n8k16.row.col.{d}.f16.f16.{c} {d}, {a}, {b}, {c}:
    row / col are the A / B layouts the fragments are in (fixed for this
    shape). Operands are register lists: a[4] and b[2] (b32, two halves each),
    c / d[4] in f32, or [2] (b32, packed halves) for an f16 accumulator.
    [EMMY] MmaSyncPtx.render calls emmy_mma_m16n8k16_{ab}_{c}(c, a, b, c);
    the wrappers in the .ref.cu inline exactly this instruction, with c as
    both d and c. The f16 accumulator form follows emmy_mma_m16n8k16_f16_f16.
    [OBS] t01 / t02: f16 operands, f32 accumulator only.
    """
    em = w.emitter
    em.comment(mma.pretty("")[0])
    shape = tuple(mma.shape)
    if shape != (16, 8, 16) or mma.ab_dtype != "f16" or mma.c_dtype not in ("f32", "f16"):
        raise NotImplementedError(f"MmaSyncPtx m{shape[0]}n{shape[1]}k{shape[2]} {mma.ab_dtype} -> {mma.c_dtype}")
    if mma.b_row_major or mma.block_scaled:
        raise NotImplementedError("MmaSyncPtx: row-major B (Volta) or block scales")

    operands = []
    for role, ref in (("c", mma.c_frag), ("a", mma.a_frag), ("b", mma.b_frag)):
        fragment, regs = fragment_registers(w, ref)
        if fragment.role != role:
            raise ValueError(f"MmaSyncPtx operand {role} is the {fragment.role} fragment {ref}")
        dtype = mma.c_dtype if role == "c" else mma.ab_dtype
        if fragment.dtype.name != dtype or len(regs) != mma_sync_nregs(role, shape, fragment.dtype):
            raise ValueError(f"MmaSyncPtx operand {role}: {ref} is {len(regs)} x {fragment.dtype.name}")
        operands.append("{" + ", ".join(regs) + "}")
    c, a, b = operands
    em.emit(f"mma.sync.aligned.m16n8k16.row.col.{mma.c_dtype}.f16.f16.{mma.c_dtype} {c}, {a}, {b}, {c}")


def emit_reg_store(w: Walker, store: RegStore) -> None:
    """Store an f32 accumulator fragment (one 16x8 C tile of the warp) to a
    global buffer, converting to its dtype.

    [PTX] m16n8k16 C layout, g = lane / 4, t = lane % 4: c0, c1 are row g,
    columns 2t and 2t + 1; c2, c3 are row g + 8, same columns.
    [EMMY] RegStore.render, unguarded path: the two columns of a row are
    contiguous, so each row is one 2-element store at
    dst[index] + row * ldm + 2t: `float2` for an f32 buffer, `__half2` packed
    with __floats2half2_rn for an f16 one. ldm is the row stride; here the
    lane's row and column are added to the row / column dims of the index,
    which flattens to the same element.
    [PTX] cvt.rn.f16x2.f32 d, a, b puts a in the upper half of d and b in the
    lower half, the one at the lower address: (c1, c0) stores c0 first.
    [OBS] t01 (f16 output) and t02's k_to__partial (f32 output): no epilogue,
    guard, atomic or swizzle, N is the last index dim.
    """
    em = w.emitter
    em.comment(store.pretty("")[0])
    if (store.fragment_layout != "m16n8k16" or store.volta_interleaved or store.epilogue is not None
            or store.extra_frags or store.m_guard is not None or store.n_guard is not None
            or store.atomic or store.swizzle != "NONE"):
        raise NotImplementedError("RegStore form: epilogue, guards, atomic, swizzle or Volta layout")
    if tuple(store.shape) != (16, 8, 16):
        raise NotImplementedError(f"RegStore of an m{store.shape[0]}n{store.shape[1]} tile")
    p = address.param(w, store.dst_buffer)
    if p is None:
        raise NotImplementedError(f"RegStore to {store.dst_buffer!r}, which is not a kernel param")
    if p.dtype not in (F32, F16):
        raise NotImplementedError(f"RegStore to a {p.dtype.name} buffer")

    fragment, regs = fragment_registers(w, store.frag)
    if fragment.role != "c" or fragment.dtype != F32 or len(regs) != 4:
        raise ValueError(f"RegStore of {store.frag}: not an f32 accumulator of 4 registers")

    index = list(store.dst_index)
    if len(index) < 2:
        raise ValueError(f"RegStore to {store.dst_buffer}: {len(index)}-d index")
    row_dim = len(index) - 2 if store.row_dim is None else store.row_dim
    col_dim = len(index) - 1 if store.col_dim is None else store.col_dim
    if col_dim != len(index) - 1:
        raise NotImplementedError("RegStore with N not on the last dim (strided column stores)")
    row_stride = 1
    for extent in p.shape[row_dim + 1:]:
        row_stride *= extent
    if store.ldm and store.ldm != row_stride:
        raise NotImplementedError(f"RegStore ldm={store.ldm}, but {store.dst_buffer}'s row stride is {row_stride}")

    lane = Var("lane")
    group = BinaryExpr("/", lane, Literal(4, "int"))
    pair_col = BinaryExpr("*", BinaryExpr("%", lane, Literal(4, "int")), Literal(2, "int"))
    for row_offset, (lo, hi) in ((0, regs[0:2]), (8, regs[2:4])):
        row = group if row_offset == 0 else BinaryExpr("+", group, Literal(row_offset, "int"))
        index[row_dim] = BinaryExpr("+", store.dst_index[row_dim], row)
        index[col_dim] = BinaryExpr("+", store.dst_index[col_dim], pair_col)
        addr = address.global_address(w, p, tuple(index))
        if p.dtype == F32:
            em.emit(f"st.global.v2.{F32.mem} [{addr}], {{{lo}, {hi}}}")
        else:
            packed = em.temp("b32")
            em.emit(f"cvt.rn.f16x2.f32 {packed}, {hi}, {lo}")
            em.emit(f"st.global.b32 [{addr}], {packed}")
