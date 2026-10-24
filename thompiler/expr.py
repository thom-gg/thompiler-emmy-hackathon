"""Index expressions -> PTX operands, and buffer element addresses.

Index expressions are small trees of Var / Literal / BinaryExpr, plus
TernaryExpr in m01 ([OBS] the open set uses only these, and only the
operators + - * / % on integer literals). They are compiled recursively;
each function returns a PTX operand: a register name or an immediate.

Index arithmetic is I32 (signed, see types.py), like the `int` arithmetic of
emmy's CUDA rendering (.ref.cu), so / and % truncate toward zero as in C.
[DOC] API.md caps a launch at 2^26 elements across all args, so every flat
element index fits in 32 bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from thompiler import ops
from thompiler.handlers.memory import SharedArray
from thompiler.load import KernelParamInfo
from thompiler.types import I32, LITERAL_TYPES, PtxType

if TYPE_CHECKING:
    from thompiler.walk import Walker


def int_expr(w: Walker, e: object) -> str:
    """A signed 32-bit integer expression, as a register or an immediate."""
    if isinstance(e, Var):
        if w.emitter.type_of(e.name) != I32:
            raise TypeError(f"{e.name} is a {w.emitter.type_of(e.name).name} value, "
                            "used in an integer expression")
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        if LITERAL_TYPES.get(e.dtype) != I32:
            raise TypeError(f"literal {e.value} of dtype {e.dtype!r} in an integer expression")
        return I32.literal(e.value)
    if isinstance(e, BinaryExpr):
        if e.op not in ops.INDEX_OPS:
            raise NotImplementedError(f"integer operator {e.op!r}")
        left, right = int_expr(w, e.left), int_expr(w, e.right)
        # If both operands are compile time known value, compute the result directly
        if _is_imm(left) and _is_imm(right):
            return str(_fold(e.op, int(left), int(right)))
        out = w.emitter.temp(I32.reg)
        instr = ops.instruction(ops.BINARY, ops.INDEX_OPS[e.op], I32)
        w.emitter.emit(f"{instr} {out}, {left}, {right}")
        return out
    if isinstance(e, TernaryExpr):
        # [EMMY] cond ? if_true : if_false. [OBS] m01 clamps a prefetch index:
        # ((_ks + 128) < 512) ? (_ks + 128) : 448. [PTX] selp.s32 d, a, b, p
        # picks a when p is true, else b.
        cond = pred_expr(w, e.cond)
        if_true, if_false = int_expr(w, e.if_true), int_expr(w, e.if_false)
        out = w.emitter.temp(I32.reg)
        w.emitter.emit(f"selp.{I32.ptx} {out}, {if_true}, {if_false}, {cond}")
        return out
    raise NotImplementedError(f"index expression {type(e).__name__}")


def pred_expr(w: Walker, e: object) -> str:
    """A boolean expression, as a predicate register (true / false per thread).

    [EMMY] Cond docstring: a condition is over axis vars (integers) and prior
    SSA names, which may be float values. A comparison is one setp, typed by
    its operands (ops.COMPARE). Mixed operands fail loudly: C would convert,
    but [DOC] CastExpr appears in no problem. && / || combine two predicates.
    """
    em = w.emitter
    if isinstance(e, BinaryExpr) and e.op in ops.COMPARE:
        t = comparison_type(w, e)
        if t == I32:
            left, right = int_expr(w, e.left), int_expr(w, e.right)
        else:
            left, right = value_operand(w, e.left, t), value_operand(w, e.right, t)
        if _is_imm(left):       # keep the first setp operand in a register
            reg = em.temp(t.reg)
            em.emit(f"mov.{t.ptx} {reg}, {left}")
            left = reg
        out = em.temp("pred")
        em.emit(f"{ops.instruction(ops.COMPARE, e.op, t)} {out}, {left}, {right}")
        return out
    if isinstance(e, BinaryExpr) and e.op in ops.LOGIC:
        left, right = pred_expr(w, e.left), pred_expr(w, e.right)
        out = em.temp("pred")
        em.emit(f"{ops.LOGIC[e.op]} {out}, {left}, {right}")
        return out
    raise NotImplementedError(f"condition {e!r}")


def comparison_type(w: Walker, e: BinaryExpr) -> PtxType:
    """The type both sides of a comparison are compared at.

    [EMMY] a float Literal has no precise type (it renders as `1.0f` and C
    converts it to the type around it), so next to a float operand it takes
    that operand's type: `x > 0.0` with an f64 x compares in f64.
    """
    left, right = operand_type(w, e.left), operand_type(w, e.right)
    if _is_float_literal(e.left) and right.family == "float":
        left = right
    if _is_float_literal(e.right) and left.family == "float":
        right = left
    if left != right:
        raise TypeError(f"comparison of mixed types: {e.pretty()}")
    return left


def _is_float_literal(e: object) -> bool:
    return isinstance(e, Literal) and e.dtype == "float"


def operand_type(w: Walker, e: object) -> PtxType:
    """Type of an expression's value: I32 for an index, F32 for a value."""
    if isinstance(e, Var):
        return w.emitter.type_of(e.name)
    if isinstance(e, Literal):
        if e.dtype in LITERAL_TYPES:
            return LITERAL_TYPES[e.dtype]
        raise NotImplementedError(f"literal of dtype {e.dtype!r}")
    if isinstance(e, BinaryExpr) and e.op in ops.INDEX_OPS:
        t = operand_type(w, e.left)
        if operand_type(w, e.right) != t:
            raise TypeError(f"arithmetic on mixed types: {e.pretty()}")
        return t
    if isinstance(e, TernaryExpr):
        t = operand_type(w, e.if_true)
        if operand_type(w, e.if_false) != t:
            raise TypeError(f"conditional with mixed branch types: {e.pretty()}")
        return t
    raise NotImplementedError(f"expression {type(e).__name__}")


def value_operand(w: Walker, e: object, t: PtxType) -> str:
    """A non-index value of a condition, of type `t`: a register or an
    immediate. [OBS] no condition computes on floats, so arithmetic is not
    supported."""
    if isinstance(e, Var):
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        return t.literal(float(e.value))
    raise NotImplementedError(f"{t.name} expression {type(e).__name__} in a condition")


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
            reg = em.temp(I32.reg)
            em.emit(f"mov.{I32.ptx} {reg}, {flat}")
            flat = reg
        out = em.temp(I32.reg)
        em.emit(f"mad.lo.{I32.ptx} {out}, {flat}, {size}, {i}")
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
        em.emit(f"mov.b64 {offset}, {int(flat) * p.dtype.nbytes}")
    else:
        em.emit(f"mul.wide.{I32.ptx} {offset}, {flat}, {p.dtype.nbytes}")
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
        byte_offset = array.offset + int(flat) * array.dtype.nbytes
        return array.symbol if byte_offset == 0 else f"{array.symbol}+{byte_offset}"
    base = em.temp(I32.reg)
    em.emit(f"mov.u32 {base}, {array.symbol}")
    if array.offset:
        array_start = em.temp(I32.reg)
        em.emit(f"add.{I32.ptx} {array_start}, {base}, {array.offset}")
        base = array_start
    addr = em.temp(I32.reg)
    em.emit(f"mad.lo.{I32.ptx} {addr}, {flat}, {array.dtype.nbytes}, {base}")
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
