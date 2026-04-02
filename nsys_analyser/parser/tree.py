from __future__ import annotations

import json
from typing import Generator, Optional, TypedDict

from ..utils import DefaultDict as ddict
from .nodes import CpuNode, CudaNode, TraceProcessNode


class TraceMetadata(TypedDict):
    processes: dict[int, str]  # pid -> process name
    thread_names: dict[int, str]  # os_tid -> thread name


StringData = dict[int, str]

# {process_id: {thread_id: [cpu_nodes]}}
CpuGroups = dict[int, dict[int, list[CpuNode]]]


class Tree:
    trees: dict[int, list[CpuNode]]  # thread_id -> root nodes
    datas: StringData  # string lookup table
    cuda_api_nodes: list[TraceProcessNode]  # cpu nodes that launched a kernel
    cuda_kernel_nodes: list[CudaNode]  # correlated gpu kernels
    meta: TraceMetadata  # process/thread naming metadata
    start: int  # earliest timestamp (ns)
    end: int  # latest timestamp (ns)

    def __init__(
        self,
        trees: dict[int, list[CpuNode]],
        datas: StringData,
        cuda_api_nodes: list[TraceProcessNode],
        cuda_kernel_nodes: list[CudaNode],
        meta: Optional[TraceMetadata] = None,
    ):
        self.trees = trees
        self.datas = datas
        self.cuda_api_nodes = cuda_api_nodes
        self.cuda_kernel_nodes = cuda_kernel_nodes
        self.meta = meta or {}

        all_roots = list(self.all_roots())
        self.start = min(r.start for r in all_roots)
        self.end = max(r.end for r in all_roots)

        root_count = sum(len(roots) for roots in self.trees.values())
        print(f"threads: {len(self.trees)}, roots: {root_count}\n")

    @property
    def time_cost(self) -> int:
        return self.end - self.start

    @property
    def processes(self) -> dict[int, str]:
        """pid -> process name."""
        return self.meta.get("processes", {})

    @property
    def thread_names(self) -> dict[int, str]:
        """globalTid -> thread name."""
        return self.meta.get("thread_names", {})

    def print_processes(self) -> None:
        """Print available processes and their threads."""
        processes = self.processes
        thread_names = self.thread_names
        if not processes:
            print("No process metadata available.")
            return

        for pid, pname in sorted(processes.items()):
            tlist = [
                (tid, name)
                for tid, name in thread_names.items()
                if tid in self.trees
            ]
            print(f"  [{pid}] {pname}  (threads: {len(tlist)})")

    def all_roots(self) -> Generator[CpuNode, None, None]:
        for roots in self.trees.values():
            for root in roots:
                yield root


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fill_tree_nodes(
    nodes: list[CpuNode],
    stacks: ddict,
    trees: ddict,
) -> None:
    for node in nodes:
        stack = stacks[node.thread]
        while len(stack) > 0 and not node.time_under(stack[-1]):
            stack.pop()
        if len(stack) == 0:
            trees[node.thread].append(node)
        else:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        stack.append(node)


def find_related(
    node: TraceProcessNode,
    correlationId_map: ddict,
) -> Optional[CudaNode]:
    cid = node.correlationId
    if correlationId_map.has(cid) and (
        "Launch" in node.text or "Memcpy" in node.text
    ):
        candidates = correlationId_map[cid]
        for idx, candidate in enumerate(candidates):
            if candidate.start > node.start:
                correlationId_map[cid].pop(idx)
                return candidate
        print(f"No correlated kernel: {node}")
    return None


def _detect_format(json_path: str) -> str:
    """Detect file format: CUPTI uses 'table' key, legacy uses 'Type'/'data' keys."""
    with open(json_path, "r") as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            if "table" in obj:
                return "cupti"
            if "Type" in obj or "data" in obj:
                return "legacy"
    raise ValueError(f"Cannot detect format of {json_path}")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def parse_meta(json_path: str) -> TraceMetadata:
    """Quick scan: parse only metadata (processes, thread names) without loading events."""
    fmt = _detect_format(json_path)
    if fmt != "cupti":
        return {"processes": {}, "thread_names": {}}

    data: dict[int, str] = {}
    processes: dict[int, str] = {}
    thread_names_raw: list[tuple[int, int]] = []

    with open(json_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            table = obj.get("table", "")
            if table == "StringIds":
                data[obj["id"]] = obj["value"]
            elif table == "PROCESSES":
                processes[obj["pid"]] = obj["name"]
            elif table == "ThreadNames":
                thread_names_raw.append((obj["globalTid"], obj["nameId"]))

    thread_names = {
        tid & 0xFFFFFF: data.get(nid, f"id:{nid}")
        for tid, nid in thread_names_raw
    }
    return {"processes": processes, "thread_names": thread_names}


def create_tree(json_path: str, filter) -> Tree:
    fmt = _detect_format(json_path)

    if fmt == "cupti":
        from .cupti.parse import create_nodes
    else:
        from .legacy.parse import create_nodes

    # 1. parse — nodes grouped by {process_id: {thread_id: [nodes]}}
    cpu_groups, cuda_groups, data, meta = create_nodes(json_path)

    # 3. apply structural filters (reduce dict)
    cpu_groups = filter(cpu_groups)

    # 5. build tree per thread
    stacks = ddict(lambda: [])
    trees = ddict(lambda: [])
    for threads in cpu_groups.values():
        for t, nodes in threads.items():
            fill_tree_nodes(nodes, stacks, trees)

    assert len(trees) > 0

    # 6. correlate per process (correlationId is process-scoped)
    cuda_api_nodes: list[TraceProcessNode] = []
    cuda_kernel_nodes: list[CudaNode] = []

    for process_id, cudas in cuda_groups.items():
        corr_map = ddict(lambda: [])
        for node in cudas:
            corr_map[node.correlationId].append(node)

        threads = cpu_groups.get(process_id, {})
        for t, nodes in threads.items():
            for node in nodes:
                if isinstance(node, TraceProcessNode):
                    related = find_related(node, corr_map)
                    if related is not None:
                        node.related = related
                        related.related = node
                        cuda_api_nodes.append(node)
                        cuda_kernel_nodes.append(related)

    return Tree(
        trees=trees,
        datas=data,
        cuda_api_nodes=cuda_api_nodes,
        cuda_kernel_nodes=cuda_kernel_nodes,
        meta=meta,
    )
