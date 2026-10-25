"""Stage 2 - Prologue: everything before the kernel body that depends only on
the kernel's name and params, not on its statements:
  - the module header        (.version / .target / .address_size)
  - the entry signature      (.visible .entry name(.param .u64 ...))
  - the param loads          (ld.param + cvta.to.global into %g_<buffer>)
  - the lane / warp ids      (when the body uses them)

TASK.md also puts the thread id and the bounds guard in the prologue. They
belong to the Tile here: the walker emits them when it meets a Tile, so we do
not assume the body is a single Tile ([OBS] it always is in the open set).

Assumptions:
  [DOC] problems are scheduled for sm_89 and PTX for sm_89 runs unchanged on
        the runner (README.md); examples/k_sum_1_reduce.ptx uses .version 8.0.
  [PTX] .version must open the module, immediately followed by .target;
        sm_89 needs PTX ISA >= 7.8. .address_size defaults to 32 when omitted.
  [DOC] buffers are .u64 pointers (TASK.md). By-value params are rejected by
        load; the signature checks it again rather than emit a wrong type.
"""

import dataclasses

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import LdmatrixLoad, RegStore, TreeHalve

from thompiler.emitter import Emitter
from thompiler.load import KernelInfo
from thompiler.types import I32

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"


def param_name(buffer: str) -> str:
    """Name of the kernel parameter carrying `buffer`'s pointer."""
    return f"param_{buffer}"


def header() -> list[str]:
    return [
        f".version {PTX_VERSION}",
        f".target {PTX_TARGET}",
        ".address_size 64",       # pointers are 64-bit; the default would be 32
    ]


def signature(kernel: KernelInfo) -> list[str]:
    for p in kernel.params:
        if p.by_value:
            raise NotImplementedError(f"by-value param {p.name!r}")
    params = ",\n".join(f"\t.param .u64 {param_name(p.name)}" for p in kernel.params)
    return [f".visible .entry {kernel.name}(", params, ")"]


def emit_param_loads(emitter: Emitter, kernel: KernelInfo) -> None:
    """Load each pointer param and convert it to a global address in
    %g_<buffer>, which Load / Write handlers read with emitter.buffer(name)."""
    for p in kernel.params:
        raw = emitter.temp("b64")
        emitter.emit(f"ld.param.u64 {raw}, [{param_name(p.name)}]")
        emitter.emit(f"cvta.to.global.u64 {emitter.define_buffer(p.name)}, {raw}")


def emit_lane_warp(emitter: Emitter, kernel: KernelInfo) -> None:
    """Define the IR's implicit `lane` and `warp` names, each only if the IR
    reads it: the thread's index in its warp, and its warp's index in the block.

    [EMMY] emmy's rendering declares `int lane = threadIdx.x & 31; int warp =
    threadIdx.x >> 5;` at function entry. (It decides by looking for the ops
    its materializer pairs with them, WarpShuffle / TreeHalve; we look for the
    reads themselves instead, which does not depend on that pairing.)
    Defined at entry, not where first read, so every path of the kernel sees them.
    They come from %tid.x, not from PTX's %warpid: [PTX] %warpid is the
    hardware warp slot, not the warp's index in the block.
    """
    reads_lane, reads_warp = _reads(kernel.body, "lane"), _reads(kernel.body, "warp")
    if not (reads_lane or reads_warp):
        return
    tid = emitter.temp(I32.reg)
    emitter.emit(f"mov.u32 {tid}, %tid.x")
    if reads_lane:
        emitter.emit(f"and.b32 {emitter.define('lane', I32.reg)}, {tid}, 31")
    if reads_warp:
        emitter.emit(f"shr.u32 {emitter.define('warp', I32.reg)}, {tid}, 5")


# Statements whose lowering reads the lane id: [EMMY] their CUDA render
# addresses each lane's part of a fragment with `threadIdx.x & 31`.
LANE_USERS = (LdmatrixLoad, RegStore)


def _reads(node: object, name: str) -> bool:
    """True when the IR under `node` reads the name `name`: a Var in any
    expression, [EMMY] TreeHalve.tid_var, which names its index variable
    as a string ([OBS] always "warp"), or for "lane" a fragment statement."""
    if isinstance(node, Var):
        return node.name == name
    if isinstance(node, TreeHalve) and node.tid_var == name:
        return True
    if name == "lane" and isinstance(node, LANE_USERS):
        return True
    if isinstance(node, (tuple, list)):
        return any(_reads(item, name) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return any(_reads(getattr(node, f.name), name) for f in dataclasses.fields(node))
    return False
