"""Instruction tables: which PTX instruction implements an operation, per
type family.

Each table maps an operation to its instruction for each type family it
supports; "{t}" is replaced by the type's PTX suffix (f32, s32, ...).
instruction() is the single place that rejects an operation a type does
not support.

[PTX] rounding: every float op that rounds is IEEE round-to-nearest (.rn).
  div / rcp / sqrt on floats need .approx or a rounding mode, and add / sub /
  mul without one may be contracted into fma by ptxas, which rounds
  differently from numpy. [OBS] p00 / p01 / p03 have rtol = atol = 0.
"""

from thompiler.types import PtxType

# Elementwise op name (Assign.op / Accum.op) -> instruction, one operand.
UNARY: dict[str, dict[str, str]] = {
    "copy":       {"float": "mov.{t}", "int": "mov.{t}"},
    "sqrt":       {"float": "sqrt.rn.{t}"},     # sqrtf
    "reciprocal": {"float": "rcp.rn.{t}"},      # 1.0f / x
}

# Elementwise op name -> instruction, two operands.
BINARY: dict[str, dict[str, str]] = {
    "add":      {"float": "add.rn.{t}", "int": "add.{t}"},
    "subtract": {"float": "sub.rn.{t}", "int": "sub.{t}"},
    "multiply": {"float": "mul.rn.{t}", "int": "mul.lo.{t}"},
    "divide":   {"float": "div.rn.{t}", "int": "div.{t}"},     # a / b, C truncation for ints
    "mod":      {"int": "rem.{t}"},                            # a % b, C truncation
    "maximum":  {"float": "max.{t}",    "int": "max.{t}"},     # fmaxf
    "minimum":  {"float": "min.{t}",    "int": "min.{t}"},     # fminf
}

# The fold op of an Accum -> its BINARY op. [EMMY] Accum docstring: op is
# "typically one of ADD / MAX / MIN / MUL". [OBS] only add and maximum.
ACCUM_OPS: tuple[str, ...] = ("add", "multiply", "maximum", "minimum")

# IR index operator (BinaryExpr.op) -> its BINARY op.
INDEX_OPS: dict[str, str] = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "%": "mod"}

# IR comparison (BinaryExpr.op) -> setp comparison. Floats follow C on NaN:
# [PTX] eq / lt / ... are ordered (false when an operand is NaN), but C's !=
# is true then, which is the unordered neu.
COMPARE: dict[str, dict[str, str]] = {
    "==": {"float": "setp.eq.{t}",  "int": "setp.eq.{t}"},
    "!=": {"float": "setp.neu.{t}", "int": "setp.ne.{t}"},
    "<":  {"float": "setp.lt.{t}",  "int": "setp.lt.{t}"},
    "<=": {"float": "setp.le.{t}",  "int": "setp.le.{t}"},
    ">":  {"float": "setp.gt.{t}",  "int": "setp.gt.{t}"},
    ">=": {"float": "setp.ge.{t}",  "int": "setp.ge.{t}"},
}

# IR logical operator -> predicate instruction (no type).
LOGIC: dict[str, str] = {"&&": "and.pred", "||": "or.pred"}


def instruction(table: dict[str, dict[str, str]], op: str, t: PtxType) -> str:
    """The instruction implementing `op` on values of type `t`, e.g.
    instruction(BINARY, "add", F32) -> "add.rn.f32"."""
    forms = table.get(op)
    if forms is None:
        raise NotImplementedError(f"operation {op!r}")
    if t.family not in forms:
        raise NotImplementedError(f"operation {op!r} on {t.name} values")
    return forms[t.family].format(t=t.ptx)
