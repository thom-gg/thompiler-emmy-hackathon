"""Handlers for leaf statements that read, compute and write values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import FlatIndex, Literal
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write

from thompiler.lowering import address, expr, ops
from thompiler.lowering.types import F32, I32, PtxType, data_type

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
    declared by an Smem (r / f / m problems): see address.memory_access.
    """
    em = w.emitter
    em.comment(load.pretty("")[0])

    width = len(load.names)
    if width not in (1, 2, 4):
        raise NotImplementedError(f"vector Load of width {width}")
    space, addr, t = address.memory_access(w, load.input, load.index)
    check_vector_bytes(width, t, f"Load from {load.input}")
    regs = [em.define(name, t.reg) for name in load.names]
    if width == 1:
        em.emit(f"ld.{space}.{t.mem} {regs[0]}, [{addr}]")
    else:
        em.emit(f"ld.{space}.v{width}.{t.mem} {{{', '.join(regs)}}}, [{addr}]")


# [PTX] a vector ld / st moves at most 16 bytes on sm_89: v4 of f32, v2 of f64.
MAX_VECTOR_BYTES = 16


def check_vector_bytes(width: int, t: PtxType, what: str) -> None:
    if width > 1 and width * t.nbytes > MAX_VECTOR_BYTES:
        raise NotImplementedError(f"{what}: {width} x {t.name} is over {MAX_VECTOR_BYTES} bytes")


LOG2_E = 1.4426950408889634         # log2(e), for exp(x) = 2^(x * log2(e))


def emit_assign(w: Walker, assign: Assign) -> None:
    """name = op(args): one elementwise operation over registers.

    [EMMY] Assign docstring: args are prior SSA names or accumulator names
    (or "$N" ports - [OBS] never seen in the open set; lookup fails loudly).
    [EMMY] dtype is optional; when None the result takes its args' type.
    Args and result share one type: [DOC] CastExpr appears in no problem.
    [OBS] every Assign is f32; ops: add copy divide exp maximum multiply
    reciprocal relu sqrt subtract.
    """
    em = w.emitter
    em.comment(assign.pretty("")[0])

    op = assign.op.name
    if len(assign.args) != assign.op.arity:
        raise ValueError(f"{op} takes {assign.op.arity} args, got {assign.args}")
    args = [em.lookup(a) for a in assign.args]
    arg_types = {em.type_of(a) for a in assign.args}
    if len(arg_types) != 1:
        raise TypeError(f"Assign {assign.name}: args of mixed types {sorted(t.name for t in arg_types)}")
    (t,) = arg_types
    result = t if assign.dtype is None else data_type(assign.dtype, f"Assign {assign.name}")
    if result != t:
        # [EMMY] dtype overrides the args' type. A change of type is a
        # conversion, which the IR spells as a copy: t02's k_to has
        # `f16 v1 = copy(acc0)` with an f32 acc0. [DOC] CastExpr appears in
        # no problem, so no other op converts.
        if op != "copy":
            raise TypeError(f"Assign {assign.name}: {op} on {t.name} args, {result.name} result")
        em.emit(f"{ops.conversion(t, result)} {em.define(assign.name, result.reg)}, {args[0]}")
        return
    dst = em.define(assign.name, t.reg)

    if op == "relu":
        # fmaxf(0.0f, x)
        em.emit(f"{ops.instruction(ops.BINARY, 'maximum', t)} {dst}, {args[0]}, {t.literal(0.0)}")
    elif op == "exp":
        # expf(x) = 2^(x * log2(e)). [PTX] ex2 exists only as ex2.approx.f32.
        # [DOC] TASK.md lists ex2.approx as the typical PTX for exp. Its error
        # is a few ulp; [OBS] every problem using exp has rtol = 1e-3.
        if t != F32:
            raise NotImplementedError(f"exp on {t.name} values")
        scaled = em.temp(t.reg)
        em.emit(f"{ops.instruction(ops.BINARY, 'multiply', t)} {scaled}, {args[0]}, {t.literal(LOG2_E)}")
        em.emit(f"ex2.approx.{t.ptx} {dst}, {scaled}")
    elif op in ops.UNARY:
        em.emit(f"{ops.instruction(ops.UNARY, op, t)} {dst}, {args[0]}")
    else:
        em.emit(f"{ops.instruction(ops.BINARY, op, t)} {dst}, {args[0]}, {args[1]}")


def emit_write(w: Walker, write: Write) -> None:
    """output[index] = values: store one value, or len(values) consecutive
    values, from registers; or atomically add one value.

    [EMMY] Write docstring: values[k] goes to index[:-1] + (index[-1] + k,),
    like a vector Load. [DOC] a grouped write is one st.global.v4 / .v2, and
    an accumulating one is red.global.add.f32 (TASK.md).
    [OBS] atomic Writes (m02) are scalar and to global memory. r / f problems
    also Write to shared arrays (acc0_smem[warp]).
    [EMMY] Write.swizzle permutes the flattened destination index of a shared
    slab "exactly like a CpAsyncCopy fill, so the LdmatrixLoad drain reads the
    slab back through the identical permutation". [OBS] always NONE so far.
    """
    em = w.emitter
    em.comment(write.pretty("")[0])

    width = len(write.values)
    if width not in (1, 2, 4) or (write.atomic and width != 1):
        raise NotImplementedError(f"Write of width {width}, atomic={write.atomic}")
    values = [em.lookup(v) for v in write.values]

    space, addr, t = address.memory_access(w, write.output, write.index, write.swizzle)
    check_vector_bytes(width, t, f"Write to {write.output}")
    value_types = {em.type_of(v) for v in write.values}
    if value_types != {t}:
        raise TypeError(f"Write to {t.name} buffer {write.output} of "
                        f"{sorted(v.name for v in value_types)} values")
    if write.atomic:
        if space != "global":
            raise NotImplementedError(f"atomic Write to shared array {write.output}")
        # red: an atomic add that does not return the old value (atom would).
        # [PTX] red.add on f16 is a different form (red.add.noftz.f16); only
        # f32 / f64 are written here.
        if t.nbytes < 4:
            raise NotImplementedError(f"atomic Write of {t.name} values")
        em.emit(f"red.global.add.{t.ptx} [{addr}], {values[0]}")
    elif width == 1:
        em.emit(f"st.{space}.{t.mem} [{addr}], {values[0]}")
    else:
        em.emit(f"st.{space}.v{width}.{t.mem} [{addr}], {{{', '.join(values)}}}")


def emit_let(w: Walker, let: Let) -> None:
    """dtype name = value: bind one pure expression to a name.

    [EMMY] Let docstring, three forms:
      - a scalar literal: "the 1 online softmax folds each element with";
        a float literal is f32 unless narrowed. [OBS] the only form used
        (f32 one = 1.0, in f02 / a01 / a02).
      - an integer index precomputed once outside a hot loop: int_expr.
      - a FlatIndex buffer offset: in no problem, fails loudly.
    """
    em = w.emitter
    em.comment(let.pretty("")[0])

    if isinstance(let.value, FlatIndex):
        raise NotImplementedError(f"Let {let.name} of a FlatIndex")
    if isinstance(let.value, Literal) and let.value.dtype == "float":
        t = data_type(let.dtype, f"Let {let.name}")
        em.emit(f"mov.{t.mem} {em.define(let.name, t.reg)}, {t.literal(float(let.value.value))}")
        return
    if expr.operand_type(w, let.value) == I32:
        # [EMMY] an untyped index takes its expression's type (`auto`).
        if let.dtype is not None and let.dtype.name != I32.name:
            raise NotImplementedError(f"Let {let.name} of integer dtype {let.dtype.name}")
        value = expr.int_expr(w, let.value)
        em.emit(f"mov.{I32.ptx} {em.define(let.name, I32.reg)}, {value}")
        return
    raise NotImplementedError(f"Let {let.name} = {let.value.pretty()}")
