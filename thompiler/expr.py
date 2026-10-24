"""Index expressions -> PTX operands, and buffer element addresses.

Index expressions are small trees of Var / Literal / BinaryExpr, plus
TernaryExpr in m01 ([OBS] the open set uses only these, and only the
operators + - * / % on integer literals). They are compiled recursively;
each function returns a PTX operand: a register name or an immediate.

Integer arithmetic is signed 32-bit, like the `int` arithmetic of emmy's
CUDA rendering (.ref.cu), so / and % truncate toward zero as in C.
[OBS] every flat element index fits in 32 bits (largest buffer: 512x512).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from thompiler.emitter import RegKind, f32_literal
from thompiler.load import KernelParamInfo
from thompiler.handlers.memory import SharedArray

if TYPE_CHECKING:
    from thompiler.walk import Walker

# IR operator -> PTX s32 instruction.
INT_OPS: dict[str, str] = {
    "+": "add.s32",
    "-": "sub.s32",
    "*": "mul.lo.s32",
    "/": "div.s32",
    "%": "rem.s32",
}


def int_expr(w: Walker, e: object) -> str:
    """A signed 32-bit integer expression, as a register or an immediate."""
    if isinstance(e, Var):
        if w.emitter.kind_of(e.name) != "b32":
            raise TypeError(f"{e.name} is a {w.emitter.kind_of(e.name)} value, "
                            "used in an integer expression")
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        if e.dtype != "int":
            raise TypeError(f"literal {e.value} of dtype {e.dtype!r} in an integer expression")
        return str(int(e.value))
    if isinstance(e, BinaryExpr):
        if e.op not in INT_OPS:
            raise NotImplementedError(f"integer operator {e.op!r}")
        left, right = int_expr(w, e.left), int_expr(w, e.right)
        # If both operands are compile time known value, compute the result directly
        if _is_imm(left) and _is_imm(right):
            return str(_fold(e.op, int(left), int(right)))
        out = w.emitter.temp("b32")
        w.emitter.emit(f"{INT_OPS[e.op]} {out}, {left}, {right}")
        return out
    if isinstance(e, TernaryExpr):
        # [EMMY] cond ? if_true : if_false. [OBS] m01 clamps a prefetch index:
        # ((_ks + 128) < 512) ? (_ks + 128) : 448. [PTX] selp.s32 d, a, b, p
        # picks a when p is true, else b.
        cond = pred_expr(w, e.cond)
        if_true, if_false = int_expr(w, e.if_true), int_expr(w, e.if_false)
        out = w.emitter.temp("b32")
        w.emitter.emit(f"selp.s32 {out}, {if_true}, {if_false}, {cond}")
        return out
    raise NotImplementedError(f"index expression {type(e).__name__}")


class SetpCmp(NamedTuple):
    """The setp comparison of one IR operator, for each operand type."""
    int: str
    f32: str


# IR comparison -> PTX setp comparison. [OBS] only == on integers is used
# (Cond). The f32 ones follow C on NaN: [PTX] eq / lt / ... are ordered
# (false when an operand is NaN), but C's != is true then: the unordered neu.
CMP_OPS: dict[str, SetpCmp] = {
    "==": SetpCmp(int="eq", f32="eq"),
    "!=": SetpCmp(int="ne", f32="neu"),
    "<": SetpCmp(int="lt", f32="lt"),
    "<=": SetpCmp(int="le", f32="le"),
    ">": SetpCmp(int="gt", f32="gt"),
    ">=": SetpCmp(int="ge", f32="ge"),
}
# IR logical op -> PTX predicate instruction.
LOGIC_OPS: dict[str, str] = {"&&": "and.pred", "||": "or.pred"}


def pred_expr(w: Walker, e: object) -> str:
    """A boolean expression, as a predicate register (true / false per thread).

    [EMMY] Cond docstring: a condition is over axis vars (integers) and prior
    SSA names, which may be f32 values. A comparison is one setp, typed by
    its operands: .s32 for integers (signed, like the C `int` comparisons of
    emmy's rendering), .f32 for floats. Mixed operands fail loudly: C would
    convert, but [DOC] CastExpr appears in no problem. && / || combine two
    predicates.
    """
    em = w.emitter
    if isinstance(e, BinaryExpr) and e.op in CMP_OPS:
        kind = operand_kind(w, e.left)
        if operand_kind(w, e.right) != kind:
            raise TypeError(f"comparison of mixed types: {e.pretty()}")
        if kind == "b32":
            left, right = int_expr(w, e.left), int_expr(w, e.right)
            cmp, ptx_type = CMP_OPS[e.op].int, "s32"
        else:
            left, right = f32_operand(w, e.left), f32_operand(w, e.right)
            cmp, ptx_type = CMP_OPS[e.op].f32, "f32"
        if _is_imm(left):       # keep the first setp operand in a register
            reg = em.temp(kind)
            em.emit(f"mov.{ptx_type} {reg}, {left}")
            left = reg
        out = em.temp("pred")
        em.emit(f"setp.{cmp}.{ptx_type} {out}, {left}, {right}")
        return out
    if isinstance(e, BinaryExpr) and e.op in LOGIC_OPS:
        left, right = pred_expr(w, e.left), pred_expr(w, e.right)
        out = em.temp("pred")
        em.emit(f"{LOGIC_OPS[e.op]} {out}, {left}, {right}")
        return out
    raise NotImplementedError(f"condition {e!r}")


def operand_kind(w: Walker, e: object) -> RegKind:
    """Register kind of an expression's value: "b32" (integer) or "f32"."""
    if isinstance(e, Var):
        return w.emitter.kind_of(e.name)
    if isinstance(e, Literal):
        if e.dtype in LITERAL_KINDS:
            return LITERAL_KINDS[e.dtype]
        raise NotImplementedError(f"literal of dtype {e.dtype!r}")
    if isinstance(e, BinaryExpr) and e.op in INT_OPS:
        kind = operand_kind(w, e.left)
        if operand_kind(w, e.right) != kind:
            raise TypeError(f"arithmetic on mixed types: {e.pretty()}")
        return kind
    if isinstance(e, TernaryExpr):
        kind = operand_kind(w, e.if_true)
        if operand_kind(w, e.if_false) != kind:
            raise TypeError(f"conditional with mixed branch types: {e.pretty()}")
        return kind
    raise NotImplementedError(f"expression {type(e).__name__}")


# Literal.dtype -> register kind. [OBS] the IRs use "int" and "float".
LITERAL_KINDS: dict[str, RegKind] = {"int": "b32", "float": "f32"}


def f32_operand(w: Walker, e: object) -> str:
    """An f32 value of a condition: a register or an f32 immediate.
    [OBS] no condition computes on floats, so arithmetic is not supported."""
    if isinstance(e, Var):
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        return f32_literal(float(e.value))
    raise NotImplementedError(f"f32 expression {type(e).__name__} in a condition")


def param(w: Walker, buffer: str) -> KernelParamInfo | None:
    """The kernel param named `buffer`, or None (e.g. a shared-memory array)."""
    return next((p for p in w.kernel.params if p.name == buffer), None)


def flat_index(w: Walker, buffer: str, shape: tuple[int, ...], index: tuple) -> str:
    """Element offset of buffer[index], flattened row-major with `shape`:
    ((i0 * s1 + i1) * s2 + i2) ... A 0-d buffer has an empty index: offset 0."""
    em = w.emitter
    if len(index) != len(shape):
        raise ValueError(f"{buffer}: {len(index)} indices for shape {shape}")
    if not index:
        return "0"

    flat = int_expr(w, index[0])
    for size, e in zip(shape[1:], index[1:]):
        i = int_expr(w, e)
        if _is_imm(flat) and _is_imm(i):
            flat = str(int(flat) * size + int(i))
            continue
        if _is_imm(flat):       # mad needs its first operand in a register
            reg = em.temp("b32")
            em.emit(f"mov.s32 {reg}, {flat}")
            flat = reg
        out = em.temp("b32")
        em.emit(f"mad.lo.s32 {out}, {flat}, {size}, {i}")
        flat = out
    return flat


def global_address(w: Walker, p: KernelParamInfo, index: tuple) -> str:
    """64-bit address of element p[index] in global memory, flattened with
    the buffer's IR shape."""
    em = w.emitter
    if not index:
        if p.shape:
            raise ValueError(f"{p.name}: no index for shape {p.shape}")
        return em.buffer(p.name)
    flat = flat_index(w, p.name, p.shape, index)

    offset, addr = em.temp("b64"), em.temp("b64")
    if _is_imm(flat):
        em.emit(f"mov.b64 {offset}, {int(flat) * p.nbytes}")
    else:
        em.emit(f"mul.wide.s32 {offset}, {flat}, {p.nbytes}")
    em.emit(f"add.s64 {addr}, {em.buffer(p.name)}, {offset}")
    return addr


def shared_address(w: Walker, array: SharedArray, index: tuple) -> str:
    """Address of element array[index] in shared memory, flattened with the
    array's extents ([EMMY] Smem docstring).

    A shared address is 32-bit: it is an offset in the block's shared window,
    and `mov.u32 r, symbol` gives the symbol's start there (nvcc's pattern for
    __shared__ arrays). The array starts array.offset bytes after its symbol
    (0 for a static array, its place in the pool for a dynamic one).
    A constant address is written as `symbol+bytes`.
    """
    em = w.emitter
    flat = flat_index(w, array.name, array.extents, index)
    if _is_imm(flat):
        byte_offset = array.offset + int(flat) * array.nbytes
        return array.symbol if byte_offset == 0 else f"{array.symbol}+{byte_offset}"
    base = em.temp("b32")
    em.emit(f"mov.u32 {base}, {array.symbol}")
    if array.offset:
        array_start = em.temp("b32")
        em.emit(f"add.s32 {array_start}, {base}, {array.offset}")
        base = array_start
    addr = em.temp("b32")
    em.emit(f"mad.lo.s32 {addr}, {flat}, {array.nbytes}, {base}")
    return addr


def _is_imm(operand: str) -> bool:
    return not operand.startswith("%")


def _fold(op: str, a: int, b: int) -> int:
    """C semantics: / and % truncate toward zero."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    q = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
    return q if op == "/" else a - q * b
