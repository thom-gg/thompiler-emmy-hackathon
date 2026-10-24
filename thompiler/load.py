"""Stage 1 - Load: read the Kernel IR and its meta.json into one KernelInfo.

Load gathers everything later stages need, so the Graph never leaves this file:
  - the entry name           (from the IR's KernelOp)
  - the params, in order     (order from meta.json's arg_order - the contract;
                              shape and dtype from the IR via graph.buffer)
  - the body                 (the statement tree to walk)
  - the dynamic smem size    (from meta.json's smem_bytes - the launch contract)
"""

import json
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.ir.stmt import Stmt
from emmy.compiler.tensor import Tensor


@dataclass
class KernelParamInfo:
    name: str
    shape: tuple[int, ...]      # IR shape, e.g. (64, 512); () for a 0-d result like sum_1
    dtype: str                  # e.g. "f32"
    nbytes: int                 # size of one element, for address arithmetic
    by_value: bool              # meta.json "scalar"; always False for now (see above)


@dataclass
class KernelInfo:
    name: str                       # PTX entry name, must match meta.json's "entry"
    params: list[KernelParamInfo]   # in meta.json's arg_order
    body: tuple[Stmt, ...]          # the KernelOp's statements (Tile, ...), untouched
    # [meta.json's smem_bytes is the dynamic shared memory the runner allocates at launch: 0 when the shared arrays are static.
    dynamic_smem_bytes: int


def load(ir_path: str | Path) -> KernelInfo:
    # Read IR using emmy-ml Graph class
    ir_path = Path(ir_path)
    graph: Graph = Graph.from_dict(json.loads(ir_path.read_text()))
    # Read the corresponding meta.json file (neighboring file, it should be in the same folder as the IR)
    # [DOC] meta.json is the contract for the entry and the parameter order.
    meta_path = ir_path.parent / "meta.json"
    meta: dict = json.loads(meta_path.read_text())

    # [DOC, implied] one KernelOp per IR file => find it.
    kernel_nodes: list[Node] = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if len(kernel_nodes) != 1:
        raise NotImplementedError(
            f"{ir_path}: expected one KernelOp, found {len(kernel_nodes)}")
    op: KernelOp = kernel_nodes[0].op


    # Find the metadata for this kernel, by looking for a kernel with the same name as what we have in our IR
    kmeta: dict | None = next((k for k in meta["kernels"] if k["entry"] == op.name), None)
    if kmeta is None:
        raise ValueError(f"kernel {op.name!r} not listed in {meta_path}")

    # Parameters needs to be in the order specified in meta.json args order, not in the order specified in the IR.
    meta_args: dict[str, dict] = {a["name"]: a for a in kmeta["args"]}
    # Build the list of params in the order we found in meta.json
    params: list[KernelParamInfo] = []
    for name in kmeta["arg_order"]:
        # By-value args (unexpected)
        by_value: bool = meta_args[name].get("scalar", False)
        if by_value:
            raise NotImplementedError(f"by-value kernel argument {name!r} in meta.json")

        # [EMMY] Graph.buffer(buf: str) -> Tensor | None: the Tensor traveling
        # under that buffer name, or None when no such buffer exists.
        tensor: Tensor | None = graph.buffer(name)
        if tensor is None:
            raise ValueError(f"arg {name!r} from arg_order has no buffer in the IR")
        if not all(d.is_static for d in tensor.shape):
            raise NotImplementedError(f"{name}: non-static shape {tensor.shape}")

        params.append(KernelParamInfo(
            name=name,
            shape=tuple(d.as_static() for d in tensor.shape),
            dtype=tensor.dtype.name,
            nbytes=tensor.dtype.nbytes,
            by_value=by_value,
        ))

    return KernelInfo(
        name=op.name,
        params=params,
        body=tuple(op.body),
        dynamic_smem_bytes=int(kmeta["smem_bytes"]),
    )
