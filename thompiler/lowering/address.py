"""Buffer element addresses: where buffer[index] is in memory.

A kernel param lives in global memory (64-bit addresses), an Smem array in
shared memory (32-bit offsets in the block's window); both flatten their
index row-major with the buffer's shape. Shared slabs may be swizzled.
Indices are compiled with expr.int_expr.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thompiler.handlers.memory import SharedArray
from thompiler.load import KernelParamInfo
from thompiler.lowering.expr import int_expr, is_imm
from thompiler.lowering.types import I32, PtxType

if TYPE_CHECKING:
    from thompiler.walk import Walker


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
        if is_imm(flat) and is_imm(i):
            flat = str(int(flat) * size + int(i))
            continue
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
    return global_element_address(w, p, flat_index(w, p.name, p.shape, index))


def global_element_address(w: Walker, p: KernelParamInfo, flat: str) -> str:
    """64-bit address of the element at flat index `flat` of buffer p. For
    accesses whose element is not a plain index tuple, such as an mma
    fragment's: the tile base plus the lane's own offset."""
    em = w.emitter
    offset, addr = em.temp("b64"), em.temp("b64")
    if is_imm(flat):
        em.emit(f"mov.b64 {offset}, {int(flat) * p.dtype.nbytes}")
    else:
        em.emit(f"mul.wide.{I32.ptx} {offset}, {flat}, {p.dtype.nbytes}")
    em.emit(f"add.s64 {addr}, {em.buffer(p.name)}, {offset}")
    return addr


def shared_address(w: Walker, array: SharedArray, index: tuple, swizzle: str = "NONE") -> str:
    """Address of element array[index] in shared memory, flattened with the
    array's extents ([EMMY] Smem docstring), then permuted by `swizzle`.

    A shared address is 32-bit: it is an offset in the block's shared window,
    and `mov.u32 r, symbol` gives the symbol's start there (nvcc's pattern for
    __shared__ arrays). The array starts array.offset bytes after its symbol
    (0 for a static array, its place in the pool for a dynamic one).
    A constant address is written as `symbol+bytes`.
    """
    return shared_element_address(w, array, flat_index(w, array.name, array.extents, index), swizzle)


def shared_element_address(w: Walker, array: SharedArray, flat: str, swizzle: str = "NONE") -> str:
    """Address of the element at flat index `flat` of a shared array, after
    `swizzle`. For accesses whose element is not a plain index tuple, such as
    an ldmatrix row: the tile base plus the lane's own offset."""
    em = w.emitter
    flat = swizzle_index(w, flat, swizzle)
    if is_imm(flat):
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


# Shared-memory swizzles: a permutation of the 16-byte chunks of each slab
# row, so the 8 rows one ldmatrix reads land in different banks. Whoever
# writes a slab and whoever reads it must apply the same mode.
# [EMMY] LDMATRIX_SWIZZLE_XOR (emmy/compiler/ir/kernel/ir.py): mode ->
# (element shift, row mask); "<mode>@<shift>" overrides the shift, for rows
# wider than 64 elements (t01's _a_smem: B128@7).
SWIZZLE_MODES: dict[str, tuple[int, int]] = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
SWIZZLE_SHIFT_SEP = "@"
# The XOR moves whole chunks of 8 elements (16 bytes of b16): `<< 3` in emmy's helper.
SWIZZLE_CHUNK_BITS = 3


def swizzle_params(mode: str) -> tuple[int, int] | None:
    """(element shift, row mask) of a swizzle mode, or None for "NONE"."""
    if mode == "NONE":
        return None
    base, sep, shift = mode.partition(SWIZZLE_SHIFT_SEP)
    if base not in SWIZZLE_MODES:
        raise NotImplementedError(f"shared-memory swizzle {mode!r}")
    default_shift, mask = SWIZZLE_MODES[base]
    return (int(shift) if sep else default_shift, mask)


def swizzle_index(w: Walker, flat: str, mode: str) -> str:
    """Element index `flat` of a slab, permuted by swizzle `mode`:

        e ^ (((e >> shift) & mask) << 3)

    [EMMY] emmy_swizzle_<mode> in the .ref.cu of t01 / t02, e.g. for B128:
    `return e ^ (((e >> 6) & 7) << 3);`. The bits above the shift pick the row,
    the mask keeps its low bits, and they are XORed into the chunk number
    (bits 3 and up), leaving the element's place in its chunk unchanged.
    """
    params = swizzle_params(mode)
    if params is None:
        return flat
    shift, mask = params
    if is_imm(flat):
        e = int(flat)
        return str(e ^ (((e >> shift) & mask) << SWIZZLE_CHUNK_BITS))
    em = w.emitter
    row, row_bits, chunk_xor, out = (em.temp(I32.reg) for _ in range(4))
    em.emit(f"shr.u32 {row}, {flat}, {shift}")
    em.emit(f"and.b32 {row_bits}, {row}, {mask}")
    em.emit(f"shl.b32 {chunk_xor}, {row_bits}, {SWIZZLE_CHUNK_BITS}")
    em.emit(f"xor.b32 {out}, {flat}, {chunk_xor}")
    return out


def memory_access(w: Walker, buffer: str, index: tuple,
                  swizzle: str = "NONE") -> tuple[str, str, PtxType]:
    """(state space, address, element type) of buffer[index] for an ld / st.

    The IR uses the same Load / Write for both memories; the buffer's name
    tells them apart: a kernel param lives in global memory, a name declared
    by an Smem in shared memory. [PTX] each space has its own instruction
    (ld.global / ld.shared) and address form (64-bit / 32-bit shared offset).
    A swizzle only permutes shared-memory slabs (swizzle_index).
    """
    p = param(w, buffer)
    if p is not None:
        if swizzle != "NONE":
            raise ValueError(f"swizzle {swizzle} on global buffer {buffer}")
        return "global", global_address(w, p, index), p.dtype
    array = w.smem.get(buffer)
    if array is not None:
        return "shared", shared_address(w, array, index, swizzle), array.dtype
    raise ValueError(f"{buffer!r} is neither a kernel param nor a declared Smem array")
