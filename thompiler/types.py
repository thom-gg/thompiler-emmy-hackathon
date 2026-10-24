"""Data types: the one place that knows how each IR dtype maps to PTX.

Handlers never spell "f32" themselves: they turn an IR dtype into a PtxType
with data_type(), which is also the single place that rejects an
unsupported dtype, and build instructions from its fields.

Supported:
  f32  the baseline: [DOC] the organizers confirmed f32 is the baseline;
       [OBS] it is the only data dtype of the open set.
  f64  buffers and values, as an extra: everything but exp (ex2 is f32
       only), WarpShuffle (shfl moves 32 bits) and 4-lane vector accesses
       (32 bytes, over the 16-byte limit). tests/problems/x03 covers it.
  i32  integer values: indices, loop counters, axis variables. They never
       come from a buffer, only from index expressions and Let.
Adding a dtype is adding a PtxType here (plus a register kind in the
emitter if it needs a new one), then the instructions it supports in ops.py.
"""

import struct
from dataclasses import dataclass
from typing import Literal

# PTX register kinds the emitter declares (.pred / .b32 / .b64 / .f32).
RegKind = Literal["pred", "b32", "b64", "f32", "f64"]
Family = Literal["float", "int"]


@dataclass(frozen=True)
class PtxType:
    name: str           # [EMMY] DataType name (emmy/compiler/dtype.py): "f32", "i32"
    family: Family      # picks the instruction form in ops.py: add.rn.f32 vs add.s32
    ptx: str            # PTX type suffix: "f32", "s32" (PTX spells signed ints sN)
    reg: RegKind        # register kind values of this type live in
    nbytes: int         # size of one element

    def literal(self, value: float) -> str:
        """`value` as a PTX immediate of this type."""
        if self.family == "int":
            return str(int(value))
        # [PTX] a float immediate is written as its IEEE bits in hex: exact,
        # unlike a decimal literal.
        if self.ptx == "f32":
            # 0f + 8 hex digits. Rounds to the nearest f32 like a C float
            # literal: 1.0 -> 0f3F800000, -1e30 -> 0fF149F2CA.
            (bits32,) = struct.unpack("<I", struct.pack("<f", value))
            return f"0f{bits32:08X}"
        if self.ptx == "f64":
            # 0d + 16 hex digits: 1.0 -> 0d3FF0000000000000.
            (bits64,) = struct.unpack("<Q", struct.pack("<d", value))
            return f"0d{bits64:016X}"
        raise NotImplementedError(f"{self.name} literal")


F32 = PtxType(name="f32", family="float", ptx="f32", reg="f32", nbytes=4)
F64 = PtxType(name="f64", family="float", ptx="f64", reg="f64", nbytes=8)
# Signed, like the C `int` of emmy's rendering: / and % truncate toward zero.
I32 = PtxType(name="i32", family="int", ptx="s32", reg="b32", nbytes=4)

# Dtypes a buffer, an Smem array or an IR value may have.
DATA_TYPES: dict[str, PtxType] = {"f32": F32, "f64": F64}

# Smem.dtype is a C type name, not an IR dtype. [OBS] only "float" is used.
SMEM_C_TYPES: dict[str, PtxType] = {"float": F32, "double": F64}

# Literal.dtype -> type. [EMMY] Literal.dtype is "int" or "float" only: a
# float literal renders as `1.0f` and C converts it to the type around it,
# so in a typed context (a comparison) it takes its operand's type instead.
LITERAL_TYPES: dict[str, PtxType] = {"int": I32, "float": F32}

# Register kind -> type of the values it holds, to type an IR name by its register.
REG_TYPES: dict[str, PtxType] = {"f32": F32, "b32": I32, "f64": F64}


def data_type(dtype: object, what: str) -> PtxType:
    """The PtxType of an IR dtype: a DataType, its name, or None.

    [EMMY] None is the legacy f32 default of Assign / Accum / Write
    (`dtype or F32` in their render). `what` names the statement for the
    error message.
    """
    if dtype is None:
        return F32
    name = str(getattr(dtype, "name", dtype))
    if name not in DATA_TYPES:
        raise NotImplementedError(f"{what}: dtype {name!r} is not supported")
    return DATA_TYPES[name]
