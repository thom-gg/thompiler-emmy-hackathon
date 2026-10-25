"""Stage 2 - Prologue: everything before the kernel body that depends only on
the kernel's name and params, not on its statements:
  - the module header        (.version / .target / .address_size)
  - the entry signature      (.visible .entry name(.param .u64 ...))
  - the param loads          (ld.param + cvta.to.global into %g_<buffer>)
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
    params = ",\n".join(f"\t.param .u64 {param_name(p.name)}" for p in kernel.params)
    return [f".visible .entry {kernel.name}(", params, ")"]


def emit_param_loads(emitter: Emitter, kernel: KernelInfo) -> None:
    """Load each pointer param and convert it to a global address in
    %g_<buffer>, which Load / Write handlers read with emitter.buffer(name)."""
    for p in kernel.params:
        raw = emitter.temp("b64")
        emitter.emit(f"ld.param.u64 {raw}, [{param_name(p.name)}]")
        emitter.emit(f"cvta.to.global.u64 {emitter.define_buffer(p.name)}, {raw}")
