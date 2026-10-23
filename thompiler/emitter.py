"""Emitter: builds the PTX text of one kernel.

The emitter knows PTX syntax, not the Kernel IR: handlers decide what to emit,
the emitter hands out registers and labels, collects lines, and assembles the
final file. It does no optimization and no liveness analysis - registers are
virtual in PTX and ptxas does the real allocation.

Registers come in two flavours:
  - named:  one per IR name, "%v_<name>" (e.g. a0 -> %v_a0), so the PTX reads
            like the IR. Writing the same register again is fine (accumulators,
            loop counters). The "v_" prefix keeps IR names from shadowing PTX
            special registers: a register declared as %tid would hide %tid.x
  - temps:  numbered, for values the IR does not name (index arithmetic,
            addresses, predicates): %r1, %rd2, %f3, %p4.
Buffer base addresses (global pointers, after cvta) are named "%g_<buffer>",
a separate namespace from IR values.

assemble() only places the pieces it is given: the module header and the
entry signature come from the prologue, so the emitter knows nothing about
the kernel itself.

Assumptions:
  [DOC] PTX rejects non-ASCII anywhere, comments included (TASK.md) - assemble
        checks it.
  [OBS] only f32 data is used, so the register kinds cover f32 values, 32-bit
        integers, 64-bit addresses and predicates. Other kinds fail loudly.
"""

from typing import Literal

RegKind = Literal["pred", "b32", "b64", "f32"]

# Register kind -> prefix of its numbered temps.
TEMP_PREFIX: dict[str, str] = {"pred": "%p", "b32": "%r", "b64": "%rd", "f32": "%f"}


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []                  # kernel body, in order
        self.shared: list[str] = []                 # module-level .shared declarations
        self.temp_count: dict[str, int] = {k: 0 for k in TEMP_PREFIX}
        self.named_regs: dict[str, str] = {}        # register ("%v_a0") -> kind ("b32")
        self.label_count = 0

    # ---- registers -------------------------------------------------------

    def temp(self, kind: RegKind) -> str:
        """A fresh numbered register for a value the IR does not name."""
        if kind not in TEMP_PREFIX:
            raise NotImplementedError(f"register kind {kind!r}")
        self.temp_count[kind] += 1
        return f"{TEMP_PREFIX[kind]}{self.temp_count[kind]}"

    def define(self, ir_name: str, kind: RegKind) -> str:
        """The register holding IR value `ir_name`, declared on first use.
        Defining it again with the same kind returns the same register."""
        return self._declare(f"%v_{ir_name}", kind)

    def lookup(self, ir_name: str) -> str:
        """The register of an IR value that was already defined."""
        reg = f"%v_{ir_name}"
        if reg not in self.named_regs:
            raise KeyError(f"IR value {ir_name!r} is read before being defined")
        return reg

    def kind_of(self, ir_name: str) -> str:
        """The register kind of an IR value, to pick typed instructions."""
        return self.named_regs[self.lookup(ir_name)]

    def define_buffer(self, buffer: str) -> str:
        """The register holding buffer `buffer`'s global base address."""
        return self._declare(f"%g_{buffer}", "b64")

    def buffer(self, buffer: str) -> str:
        reg = f"%g_{buffer}"
        if reg not in self.named_regs:
            raise KeyError(f"buffer {buffer!r} has no address register")
        return reg

    def _declare(self, reg: str, kind: RegKind) -> str:
        if kind not in TEMP_PREFIX:
            raise NotImplementedError(f"register kind {kind!r}")
        known = self.named_regs.setdefault(reg, kind)
        if known != kind:
            raise TypeError(f"{reg} declared as .{known}, now used as .{kind}")
        return reg

    # ---- body text -------------------------------------------------------

    def label(self, hint: str) -> str:
        """A fresh unique label name, e.g. label("loop") -> "LOOP_1"."""
        self.label_count += 1
        return f"{hint.upper()}_{self.label_count}"

    def emit(self, instr: str) -> None:
        """One instruction, without the trailing ';'."""
        self.lines.append(f"\t{instr};")

    def place_label(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def comment(self, text: str) -> None:
        self.lines.append(f"\t// {text}")

    def declare_shared(self, decl: str) -> None:
        """A module-level declaration, e.g. ".shared .align 4 .b8 acc0_smem[16]"."""
        self.shared.append(f"{decl};")

    # ---- final file ------------------------------------------------------

    def assemble(self, header: list[str], signature: list[str]) -> str:
        """The whole PTX file: header, .shared declarations, signature, then
        the function body. Called last: the .reg declarations need the final
        register counts."""
        regs = [f"\t.reg .{kind} {TEMP_PREFIX[kind]}<{n + 1}>;"
                for kind, n in self.temp_count.items() if n > 0]
        for kind in TEMP_PREFIX:
            names = [r for r, k in self.named_regs.items() if k == kind]
            if names:
                regs.append(f"\t.reg .{kind} {', '.join(names)};")

        ptx = "\n".join([
            *header,
            "",
            *self.shared,
            "",
            *signature,
            "{",
            *regs,
            "",
            *self.lines,
            "}",
            "",
        ])
        if not ptx.isascii():
            bad = sorted({c for c in ptx if not c.isascii()})
            raise ValueError(f"PTX must be ASCII, found {bad}")
        return ptx
