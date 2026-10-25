"""IR expressions -> PTX operands.

Each compiling function (int_expr, pred_expr, fp_operand) emits the PTX
that computes the expression's value, if any is needed, and returns the
operand holding that value, for the caller to use in its own instruction:
  - a register ("%r12", "%p3", "%v_a0") when the value is only known at run
    time; the instructions computing it were emitted first;
  - an immediate ("64", "0f3F800000") when it is known at compile time (a
    literal, an unrolled loop variable, or + - * / % between two of them,
    folded): nothing is emitted.

comparison_type and operand_type emit nothing: they only type an expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var

from thompiler.lowering import ops
from thompiler.lowering.types import I32, LITERAL_TYPES, PtxType

if TYPE_CHECKING:
    from thompiler.walk import Walker


def int_expr(w: Walker, e: object) -> str:
    """A signed 32-bit integer expression, as a register or an immediate."""
    if isinstance(e, Var) and e.name in w.consts:
        return I32.literal(w.consts[e.name])     # an unrolled loop's variable
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
        if is_imm(left) and is_imm(right):
            return str(_fold(e.op, int(left), int(right)))
        out = w.emitter.temp(I32.reg)
        instr = ops.instruction(ops.BINARY, ops.INDEX_OPS[e.op], I32)
        w.emitter.emit(f"{instr} {out}, {left}, {right}")
        return out
    if isinstance(e, TernaryExpr):
        cond = pred_expr(w, e.cond)
        if_true, if_false = int_expr(w, e.if_true), int_expr(w, e.if_false)
        out = w.emitter.temp(I32.reg)
        w.emitter.emit(f"selp.{I32.ptx} {out}, {if_true}, {if_false}, {cond}")
        return out
    raise NotImplementedError(f"index expression {type(e).__name__}")


def pred_expr(w: Walker, e: object) -> str:
    """A boolean expression, as a predicate register (true / false per thread).
    """
    em = w.emitter
    if isinstance(e, BinaryExpr) and e.op in ops.COMPARE:
        t = comparison_type(w, e)
        if t.family == "int":
            left, right = int_expr(w, e.left), int_expr(w, e.right)
        else:
            left, right = fp_operand(w, e.left, t), fp_operand(w, e.right, t)
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
    """The type both sides of a comparison are compared at: they must have
    the same one."""
    left, right = operand_type(w, e.left), operand_type(w, e.right)
    if left != right:
        raise TypeError(f"comparison of mixed types: {e.pretty()}")
    return left


def operand_type(w: Walker, e: object) -> PtxType:
    """Type of an expression's value: its register's type for a Var (I32 for
    an unrolled loop variable), LITERAL_TYPES for a literal, and its
    operands' common type for arithmetic and conditionals."""
    if isinstance(e, Var) and e.name in w.consts:
        return I32
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


def fp_operand(w: Walker, e: object, t: PtxType) -> str:
    """A floating-point operand of a condition, of type `t` (f16, f32 or
    f64): a register or an immediate. [OBS] no condition computes on
    floating-point values, so arithmetic is not supported."""
    if t.family != "float":
        raise TypeError(f"fp_operand of a {t.name} value")
    if isinstance(e, Var):
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        return t.literal(float(e.value))
    raise NotImplementedError(f"{t.name} expression {type(e).__name__} in a condition")


def add_ints(w: Walker, a: str, b: str) -> str:
    """a + b for two I32 operands, folded when both are immediates."""
    if is_imm(a) and is_imm(b):
        return str(int(a) + int(b))
    out = w.emitter.temp(I32.reg)
    w.emitter.emit(f"add.{I32.ptx} {out}, {a}, {b}")
    return out


def is_imm(operand: str) -> bool:
    """Checks wether this operand is an imediate value"""
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
