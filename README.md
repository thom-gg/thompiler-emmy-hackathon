# the thompiler: Emmy Hackathon 2026 (Kernel IR => PTX)

My submission for the Emmy Hackathon (https://riftstack.ai/hackathon) ([github](https://github.com/cloudrift-ai/emmy-hackathon-2026))

`compile.py` reads one Emmy Kernel IR file and the meta.json file supposed to be in the same folder, and prints PTX on stdout, with no nvcc and no Emmy renderer in between:

```bash
python3 compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
```

```bash
python3 scripts/check_all.py          # compile every problem locally
python3 scripts/check_all.py --gpu    # compile, then run each one on the GPU runner
```

`check_all.py` covers `problems/` (the open set) and `tests/problems/` (additional generated problems),
and stops at the first compile or GPU failure.

## What my compiler supports

**The open set: all 15 problems pass** on the runner GPU (RTX 5090), categories 1 to 5 (pointwise, reductions, matmul, fused and attention-like)

**Beyond the open set**, it also supports some more features:

- **Tensor cores instructions** (problems t01 and t02): `RegFragment`, `LdmatrixLoad`, `MmaSyncPtx` and `RegStore`. It only supports the `m16n8k16` shape with either f16 or f32 accumulator, because it was what was most easily emitted by Emmy and I figured it would be enough for a proof of concept.
- **Shared-memory swizzle**: the B128, B64, B32 and `@shift` modes Emmy uses
  to avoid bank conflicts in `ldmatrix`, on every `cp.async`, `Write` and
  `ldmatrix` that carries one.
- **Additional datatypes** (problems x01 and x02): i made it easy to support new datatypes by keeping related methods in `types.py` and `ops.py`, and illustrated that with f64 support.


`submit.py` casts every output to float32 before comparing it, so f64 results
are checked to f32 precision only.

## Repository layout

```
compile.py                 entry point: load -> prologue -> walk -> assemble
thompiler/
  load.py                  1) read the IR and its meta.json
  prologue.py              2) module header, entry signature, param loads
  walk.py                  3) the Walker: dispatches each IR statement to its handler
  emitter.py               builds the PTX text: registers, labels, lines, assemble()
  handlers/                one module per family of IR statements
    blocks.py              Tile, Loop, StridedLoop, Cond
    compute.py             Load, Write, Assign, Let
    reduce.py              Accum, Init, WarpShuffle, TreeHalve
    memory.py              Smem (static arrays or a dynamic pool), Sync
    cp_async.py            CpAsyncCopy / Commit / Wait
    mma.py                 RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore
  lowering/                building blocks the handlers share: a piece of IR -> PTX
    types.py               IR dtype -> PtxType (f16, f32, f64, i32)
    ops.py                 IR operation -> PTX instruction, per type family
    expr.py                IR expression -> PTX operand
    address.py             buffer[index] -> memory address (global / shared, swizzle)
scripts/check_all.py       compile (and with --gpu, run) every problem
tests/                     torch expressions and the problems generated from them
```

## How the compiler works

1. **Load** reads the IR and, from the `meta.json` next to it (the launch
   contract), the argument order and the dynamic shared memory size.
2. **Prologue** emits the module header and loads each pointer parameter
   (`ld.param` + `cvta.to.global`). The thread id and bounds guard, which
   `TASK.md` suggests putting here, are in the `Tile` handler instead: they
   belong to the tile and its axes.
3. **Walk**: the Walker calls each statement's handler by its class, and
   carries the context they need (the emitter, the declared shared arrays,
   the tensor-core fragments, the values of unrolled loops). Block handlers
   recurse with `walker.body(...)`.
4. **Assemble**: the emitter only produces the text at the end, because the
   `.reg` declarations and `.shared` arrays go at the top but are only known
   once the whole IR has been walked.

The handlers build their PTX from `lowering/`: `expr.py` compiles an IR
expression and returns the operand holding its value (a register, or an
immediate when it is known at compile time); `address.py` computes where
`buffer[index]` is; `types.py` and `ops.py` centralize the supported types
and operations, so supporting another type is a matter of adding it there.

## Some design choices

- **Readable PTX**: IR values get named registers (`a0` lives in `%v_a0`,
  a global memory parameter buffer's address in `%g_<buffer>`, a fragment in `%x_<frag>_<member>_<slot>`),
  and IR statements are emitted with their pretty-printed form as a comment. It shouldn't change anything about the behavior / performance of the generated code, but it makes it easier to read / debug the PTX.
- **Type hints**: it also just makes it easier to read doesn't it ? i'm kind of a typescript guy 👉👈
- **Fail loudly** rather than guess: never let an unexpected case slip through "just to see" if the generated ptx would work ; raise an exception directly during compilation so the issue can be addressed.


## Interesting handlers

- `emit_load` (`handlers/compute.py`): one IR statement, different types and
  address spaces, through `address.py` and `types.py`.
- `emit_smem` (`handlers/memory.py`): shared arrays are static, or regions of
  one dynamic pool at aligned offsets, depending on the launch contract.
- The swizzle: `emit_write` gets it through `address.memory_access`,
  `emit_cp_async_copy` through `address.shared_address`, and `LdmatrixLoad`
  adds each lane's offset before swizzling.
- `emit_ldmatrix_load` / `emit_mma_sync` (`handlers/mma.py`): fragment
  families become named registers, and the loops indexing them are unrolled
  at compile time, since PTX registers cannot be indexed at run time.
