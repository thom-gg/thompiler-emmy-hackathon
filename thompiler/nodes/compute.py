"""Handlers for leaf statements that read, compute and write values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.stmt.leaves import Assign, Load, Write

from thompiler import expr

if TYPE_CHECKING:
    from thompiler.walk import Walker


def emit_load(w: Walker, load: Load) -> None:
    """names = load input[index]: read one value, or len(names) consecutive
    values, into the registers of names.

    [EMMY] Load docstring: a vector load (len(names) > 1) reads lane k from
    index[:-1] + (index[-1] + k,), i.e. consecutive elements of the last
    dimension. [DOC] it is one ld.global.v2 / .v4 (TASK.md).
    [OBS] vector loads appear only in a01 (width 2). A vector access must be
    aligned to its total size; we rely on emmy's vectorize pass for that.
    [OBS] the buffer is a kernel param (global memory) or a shared array
    declared by an Smem (r / f / m problems): see memory_access.
    """
    em = w.emitter
    em.comment(load.pretty("")[0])

    width = len(load.names)
    if width not in (1, 2, 4):
        raise NotImplementedError(f"vector Load of width {width}")
    space, addr = memory_access(w, load.input, load.index)
    regs = [em.define(name, "f32") for name in load.names]
    if width == 1:
        em.emit(f"ld.{space}.f32 {regs[0]}, [{addr}]")
    else:
        em.emit(f"ld.{space}.v{width}.f32 {{{', '.join(regs)}}}, [{addr}]")


def memory_access(w: Walker, buffer: str, index: tuple) -> tuple[str, str]:
    """(state space, address) of buffer[index] for an ld / st.

    The IR uses the same Load / Write for both memories; the buffer's name
    tells them apart: a kernel param lives in global memory, a name declared
    by an Smem in shared memory. [PTX] each space has its own instruction
    (ld.global / ld.shared) and address form (64-bit / 32-bit shared offset).
    """
    p = expr.param(w, buffer)
    if p is not None:
        if p.dtype != "f32":
            raise NotImplementedError(f"access to {buffer} of dtype {p.dtype}")
        return "global", expr.global_address(w, p, index)
    array = w.smem.get(buffer)
    if array is not None:
        if array.dtype != "f32":
            raise NotImplementedError(f"access to {buffer} of dtype {array.dtype}")
        return "shared", expr.shared_address(w, array, index)
    raise ValueError(f"{buffer!r} is neither a kernel param nor a declared Smem array")


# Elementwise op name -> PTX f32 instruction, for ops that are one instruction.
# Every rounding op is IEEE round-to-nearest (.rn):
#   [PTX] div, rcp and sqrt on .f32 need .approx or a rounding mode.
#   [PTX] add / sub / mul without a rounding modifier may be contracted into
#         fma by ptxas, which rounds differently from numpy; an explicit .rn
#         prevents it. [OBS] p00 / p01 / p03 have rtol = atol = 0.
UNARY_F32: dict[str, str] = {
    "copy": "mov.f32",
    "sqrt": "sqrt.rn.f32",          # sqrtf
    "reciprocal": "rcp.rn.f32",     # 1.0f / x
}
BINARY_F32: dict[str, str] = {
    "add": "add.rn.f32",
    "subtract": "sub.rn.f32",
    "multiply": "mul.rn.f32",
    "divide": "div.rn.f32",         # a / b
    "maximum": "max.f32",           # fmaxf
}
LOG2_E = "0f3FB8AA3B"               # log2(e) = 1.4426950 as an f32 hex literal


def emit_assign(w: Walker, assign: Assign) -> None:
    """name = op(args): one elementwise operation over registers.

    [EMMY] Assign docstring: args are prior SSA names or accumulator names
    (or "$N" ports - [OBS] never seen in the open set; lookup fails loudly).
    [EMMY] dtype is optional; when None the result takes its args' type.
    [OBS] every Assign is f32; ops: add copy divide exp maximum multiply
    reciprocal relu sqrt subtract.
    """
    em = w.emitter
    em.comment(assign.pretty("")[0])

    op = assign.op.name
    if len(assign.args) != assign.op.arity:
        raise ValueError(f"{op} takes {assign.op.arity} args, got {assign.args}")
    args = [em.lookup(a) for a in assign.args]
    kinds = {em.kind_of(a) for a in assign.args}
    dtype = assign.dtype.name if assign.dtype is not None else None
    if kinds != {"f32"} or dtype not in (None, "f32"):
        raise NotImplementedError(f"Assign {op} on {kinds}, dtype {dtype}")
    dst = em.define(assign.name, "f32")

    if op in UNARY_F32:
        em.emit(f"{UNARY_F32[op]} {dst}, {args[0]}")
    elif op in BINARY_F32:
        em.emit(f"{BINARY_F32[op]} {dst}, {args[0]}, {args[1]}")
    elif op == "relu":
        # fmaxf(0.0f, x)
        em.emit(f"max.f32 {dst}, {args[0]}, 0f00000000")
    elif op == "exp":
        # expf(x) = 2^(x * log2(e)). [PTX] ex2 on .f32 exists only as .approx.
        # [DOC] TASK.md lists ex2.approx as the typical PTX for exp. Its error
        # is a few ulp; [OBS] every problem using exp has rtol = 1e-3.
        scaled = em.temp("f32")
        em.emit(f"mul.rn.f32 {scaled}, {args[0]}, {LOG2_E}")
        em.emit(f"ex2.approx.f32 {dst}, {scaled}")
    else:
        raise NotImplementedError(f"elementwise op {op!r}")


def emit_write(w: Walker, write: Write) -> None:
    """output[index] = values: store one value, or len(values) consecutive
    values, from registers; or atomically add one value.

    [EMMY] Write docstring: values[k] goes to index[:-1] + (index[-1] + k,),
    like a vector Load. [DOC] a grouped write is one st.global.v4 / .v2, and
    an accumulating one is red.global.add.f32 (TASK.md).
    [OBS] atomic Writes (m02) are scalar and to global memory; swizzle is
    always NONE. r / f problems also Write to shared arrays (acc0_smem[warp]).
    """
    em = w.emitter
    em.comment(write.pretty("")[0])

    if str(write.swizzle) != "NONE":
        raise NotImplementedError(f"Write with swizzle {write.swizzle}")
    width = len(write.values)
    if width not in (1, 2, 4) or (write.atomic and width != 1):
        raise NotImplementedError(f"Write of width {width}, atomic={write.atomic}")
    values = [em.lookup(v) for v in write.values]
    kinds = {em.kind_of(v) for v in write.values}
    if kinds != {"f32"}:
        raise NotImplementedError(f"Write of {kinds} values")

    space, addr = memory_access(w, write.output, write.index)
    if write.atomic:
        if space != "global":
            raise NotImplementedError(f"atomic Write to shared array {write.output}")
        # red: an atomic add that does not return the old value (atom would).
        em.emit(f"red.global.add.f32 [{addr}], {values[0]}")
    elif width == 1:
        em.emit(f"st.{space}.f32 [{addr}], {values[0]}")
    else:
        em.emit(f"st.{space}.v{width}.f32 [{addr}], {{{', '.join(values)}}}")
