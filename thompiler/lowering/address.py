"""Where buffer[index] is in memory: the PTX computing the address of a
buffer element, for the handlers' ld / st, cp.async and ldmatrix.

Two memories, two address forms ([PTX]):
  - a kernel param is in global memory: a 64-bit address, the buffer's base
    register (%g_<buffer>) plus the element's byte offset;
  - an Smem array is in shared memory: a 32-bit offset in the block's shared
    window, from the array's symbol. A shared slab may also be swizzled.
Both flatten a multi-dimensional index row-major with the buffer's shape
(flat_index), then scale it by the element size. Like expr.py, each function
emits the instructions it needs and returns the operand holding the result:
a register, or a constant address.

Entry points:
  memory_access            buffer name + index -> (space, address, type)
  global_address           kernel param + index -> 64-bit address
  shared_address           Smem array + index (+ swizzle) -> shared address
  shared_element_address   Smem array + flat index (+ swizzle) -> shared address,
                           for an element that is not a plain index (ldmatrix's
                           tile base plus the lane's offset)
  flat_index, param        the two steps the above are built from
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
    flat = flat_index(w, p.name, p.shape, index)

    offset, addr = em.temp("b64"), em.temp("b64")
    if is_imm(flat):
        em.emit(f"mov.b64 {offset}, {int(flat) * p.dtype.nbytes}")
    else:
        em.emit(f"mul.wide.{I32.ptx} {offset}, {flat}, {p.dtype.nbytes}")
    em.emit(f"add.s64 {addr}, {em.buffer(p.name)}, {offset}")
    return addr


def shared_address(w: Walker, array: SharedArray, index: tuple, swizzle: str = "NONE") -> str:
    """Address of element array[index] in shared memory, flattened with the
    array's extents then permuted by `swizzle`.
    """
    return shared_element_address(w, array, flat_index(w, array.name, array.extents, index), swizzle)


def shared_element_address(w: Walker, array: SharedArray, flat: str, swizzle: str = "NONE") -> str:
    """Address of the element at flat index `flat` of a shared array, after
    `swizzle`. For accesses whose element is not a plain index tuple, such as
    an ldmatrix row: the tile base plus the lane's own offset."""
    em = w.emitter
    flat = _swizzle_index(w, flat, swizzle)
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
# wider than 64 element
SWIZZLE_MODES: dict[str, tuple[int, int]] = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}
SWIZZLE_SHIFT_SEP = "@"
# The XOR moves whole chunks of 8 elements (16 bytes of b16): `<< 3` in emmy's helper.
SWIZZLE_CHUNK_BITS = 3


def _swizzle_params(mode: str) -> tuple[int, int] | None:
    """(element shift, row mask) of a swizzle mode, or None for "NONE"."""
    if mode == "NONE":
        return None
    base, sep, shift = mode.partition(SWIZZLE_SHIFT_SEP)
    if base not in SWIZZLE_MODES:
        raise NotImplementedError(f"shared-memory swizzle {mode!r}")
    default_shift, mask = SWIZZLE_MODES[base]
    return (int(shift) if sep else default_shift, mask)


def _swizzle_index(w: Walker, flat: str, mode: str) -> str:
    """Element index `flat` of a slab, permuted by swizzle `mode`:

        e ^ (((e >> shift) & mask) << 3)
    """
    params = _swizzle_params(mode)
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
    by an Smem in shared memory.
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
