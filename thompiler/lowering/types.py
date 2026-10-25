"""Data types: how each IR dtype maps to PTX.

An IR dtype becomes a PtxType through data_type() (buffers, IR values) or
SMEM_C_TYPES (Smem arrays, typed by a C name); each rejects a dtype we do
not support. Handlers build their instructions from the PtxType's fields,
except the mma.sync path, whose instruction forms are fixed and spell their
f16 / f32 types themselves (handlers/mma.py).

Supported:
  f32  the baseline: it is the only data dtype of the open set.
  f64  buffers and values: everything but exp (ex2 is f32 only),
       WarpShuffle (shfl moves 32 bits)
  f16  buffers, values and mma fragments: loads, stores, copies and
       conversions to / from f32 and f64 only (ops.HALF_OPS): no
       arithmetic, comparison, literal or accumulation.
  i32  integer values: axis variables, loop counters, lane / warp, and what
       index expressions and Let compute from them. Never a buffer element.
"""

import struct
from dataclasses import dataclass
from typing import Literal

# PTX register kinds the emitter declares (.pred / .b16 / .b32 / .b64 / .f32 / .f64).
RegKind = Literal["pred", "b16", "b32", "b64", "f32", "f64"]
Family = Literal["float", "int"]


@dataclass(frozen=True)
class PtxType:
    name: str           # DataType name (emmy/compiler/dtype.py): "f32", "i32"
    family: Family      # picks the instruction template in ops.py: "add.rn.{t}" vs "add.{t}"
    ptx: str            # PTX type suffix filling {t}: "f32", "s32" (PTX spells signed ints sN)
    reg: RegKind        # register kind values of this type live in
    nbytes: int         # size of one element
    # Suffix of the instructions that only move the value (ld / st / mov). It
    # is equal to the ptx field, except for f16
    mem: str

    def literal(self, value: float) -> str:
        """`value` as a PTX immediate of this type."""
        if self.family == "int":
            return str(int(value))
        # a float immediate is written as its IEEE bits in hex
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


F16 = PtxType(name="f16", family="float", ptx="f16", reg="b16", nbytes=2, mem="b16")
F32 = PtxType(name="f32", family="float", ptx="f32", reg="f32", nbytes=4, mem="f32")
F64 = PtxType(name="f64", family="float", ptx="f64", reg="f64", nbytes=8, mem="f64")
I32 = PtxType(name="i32", family="int", ptx="s32", reg="b32", nbytes=4, mem="s32")

# IR dtype name -> type, for buffers and IR values (data_type). i32 is not
# one: no buffer holds integers.
DATA_TYPES: dict[str, PtxType] = {"f16": F16, "f32": F32, "f64": F64}

# Smem.dtype is a C type name, not an IR dtype.
SMEM_C_TYPES: dict[str, PtxType] = {"__half": F16, "float": F32, "double": F64}

# Literal.dtype -> type.
LITERAL_TYPES: dict[str, PtxType] = {"int": I32, "float": F32}

# Register kind -> type of the values it holds, to type an IR name by its register.
REG_TYPES: dict[str, PtxType] = {"b16": F16, "f32": F32, "b32": I32, "f64": F64}


def data_type(dtype: object, what: str) -> PtxType:
    """The PtxType of an IR dtype: a DataType, its name, or None.
    """
    if dtype is None:
        return F32
    name = str(getattr(dtype, "name", dtype))
    if name not in DATA_TYPES:
        raise NotImplementedError(f"{what}: dtype {name!r} is not supported")
    return DATA_TYPES[name]
