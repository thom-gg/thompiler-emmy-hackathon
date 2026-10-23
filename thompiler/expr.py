"""Index expressions -> PTX operands, and buffer element addresses.

Index expressions are small trees of Var / Literal / BinaryExpr ([OBS] the
open set uses only these, plus one TernaryExpr in m01, and only the
operators + - * / % on integer literals). They are compiled recursively;
each function returns a PTX operand: a register name or an immediate.

Integer arithmetic is signed 32-bit, like the `int` arithmetic of emmy's
CUDA rendering (.ref.cu), so / and % truncate toward zero as in C.
[OBS] every flat element index fits in 32 bits (largest buffer: 512x512).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

from thompiler.load import KernelParamInfo

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
        return w.emitter.lookup(e.name)
    if isinstance(e, Literal):
        if e.dtype != "int":
            raise NotImplementedError(f"literal of dtype {e.dtype!r} in an index")
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
    raise NotImplementedError(f"index expression {type(e).__name__}")


def param(w: Walker, buffer: str) -> KernelParamInfo | None:
    """The kernel param named `buffer`, or None (e.g. a shared-memory array)."""
    return next((p for p in w.kernel.params if p.name == buffer), None)


def global_address(w: Walker, p: KernelParamInfo, index: tuple) -> str:
    """64-bit address of element p[index] in global memory.

    The index has one expression per dimension; it is flattened row-major
    with the buffer's IR shape: ((i0 * s1 + i1) * s2 + i2) ...
    A 0-d buffer (shape ()) has an empty index: its address is the base.
    """
    em = w.emitter
    if len(index) != len(p.shape):
        raise ValueError(f"{p.name}: {len(index)} indices for shape {p.shape}")
    if not index:
        return em.buffer(p.name)

    flat = int_expr(w, index[0])
    for size, e in zip(p.shape[1:], index[1:]):
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

    offset, addr = em.temp("b64"), em.temp("b64")
    if _is_imm(flat):
        em.emit(f"mov.b64 {offset}, {int(flat) * p.nbytes}")
    else:
        em.emit(f"mul.wide.s32 {offset}, {flat}, {p.nbytes}")
    em.emit(f"add.s64 {addr}, {em.buffer(p.name)}, {offset}")
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
