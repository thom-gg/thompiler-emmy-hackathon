"""Stage 2 - Prologue: everything before the kernel body that depends only on
the kernel's name and params, not on its statements:
  - the module header        (.version / .target / .address_size)
  - the entry signature      (.visible .entry name(.param .u64 ...))
  - the param loads          (ld.param + cvta.to.global into %g_<buffer>)

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

from thompiler.emitter import Emitter
from thompiler.load import KernelInfo

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
