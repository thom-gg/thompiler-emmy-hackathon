"""Instruction tables: which PTX instruction implements an operation, per
type family.

Each table maps an operation to its instruction for each type family it
supports; "{t}" is replaced by the type's PTX suffix (f32, s32, ...) and
"{mem}" by its move suffix (types.PtxType.mem). instruction() is the single
place that rejects an operation a type does not support; conversion() does
the same for a change of type.

[PTX] rounding: every float op that rounds is IEEE round-to-nearest (.rn).
  div / rcp / sqrt on floats need .approx or a rounding mode, and add / sub /
  mul without one may be contracted into fma by ptxas, which rounds
  differently from numpy. [OBS] p00 / p01 / p03 have rtol = atol = 0.
"""

from thompiler.lowering.types import PtxType

# Elementwise op name (Assign.op / Accum.op) -> instruction, one operand.
UNARY: dict[str, dict[str, str]] = {
    "copy":       {"float": "mov.{mem}", "int": "mov.{mem}"},
    "sqrt":       {"float": "sqrt.rn.{t}"},     # sqrtf
    "reciprocal": {"float": "rcp.rn.{t}"},      # 1.0f / x
    "negative":   {"float": "neg.{t}", "int": "neg.{t}"},   # 0.0f - x (but -0.0 for x = 0)
    "abs":        {"float": "abs.{t}", "int": "abs.{t}"},   # fabsf
    "rsqrt":      {"f32": "rsqrt.approx.f32"},  # rsqrtf
    "tanh":       {"f32": "tanh.approx.f32"},   # tanhf
    "sin":        {"f32": "sin.approx.f32"},    # sinf
    "cos":        {"f32": "cos.approx.f32"},    # cosf
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

# The fold op of an Accum -> its BINARY op.
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


# Operations on f16 values. for now f16 values are only moved and
# converted, and computed on in f32.
HALF_OPS: frozenset[str] = frozenset({"copy"})


def instruction(table: dict[str, dict[str, str]], op: str, t: PtxType) -> str:
    """The instruction implementing `op` on values of type `t`, e.g.
    instruction(BINARY, "add", F32) -> "add.rn.f32"."""
    forms = table.get(op)
    if forms is None:
        raise NotImplementedError(f"operation {op!r}")
    form = forms.get(t.name, forms.get(t.family))
    if form is None or (t.nbytes == 2 and op not in HALF_OPS):
        raise NotImplementedError(f"operation {op!r} on {t.name} values")
    return form.format(t=t.ptx, mem=t.mem)


# (source, destination) dtype names -> conversion. [PTX] cvt needs a rounding
# mode when the conversion can lose precision (narrowing: .rn, IEEE nearest),
# and forbids one otherwise (widening is exact).
CONVERT: dict[tuple[str, str], str] = {
    ("f32", "f16"): "cvt.rn.f16.f32",
    ("f64", "f16"): "cvt.rn.f16.f64",
    ("f64", "f32"): "cvt.rn.f32.f64",
    ("f16", "f32"): "cvt.f32.f16",
    ("f16", "f64"): "cvt.f64.f16",
    ("f32", "f64"): "cvt.f64.f32",
}


def conversion(src: PtxType, dst: PtxType) -> str:
    """The instruction converting a `src` value to `dst`, e.g.
    conversion(F32, F16) -> "cvt.rn.f16.f32"."""
    if (src.name, dst.name) not in CONVERT:
        raise NotImplementedError(f"conversion from {src.name} to {dst.name}")
    return CONVERT[(src.name, dst.name)]
