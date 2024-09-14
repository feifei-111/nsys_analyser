import enum
import json

from ..utils import DefaultDict as ddict
from .event_node import *


class FilterResult(enum.Enum):
    SAVE = 0
    DROP = 1
    BREAK = 2


class Tree:
    def __init__(self, **kwargs):
        self.trees = kwargs["trees"]
        self.datas = kwargs["datas"]
        self.cpu_nodes = kwargs["cpu_nodes"]
        self.cuda_api_nodes = kwargs["cuda_api_nodes"]
        self.cuda_kernel_nodes = kwargs["cuda_kernel_nodes"]
        self.main_thread = kwargs["main_thread"]

        self.main_roots = self.trees[self.main_thread]
        self.start = self.trees[self.main_thread][0].start
        self.end = self.trees[self.main_thread][-1].end

        print(
            f"threads: {len(self.trees)}, main_thread_trees: {len(self.main_roots)}\n"
        )

    @property
    def time_cost(self):
        return self.end - self.start

    def all_roots(self):
        for k, roots in self.trees.items():
            for root in roots:
                yield root


def create_nodes(json_path):
    datas = []
    cudas = []
    cpus = []
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            nsys_json = json.loads(line)

            if "Type" not in nsys_json:
                datas.append(nsys_json)
            else:
                event_type = nsys_json["Type"]
                if event_type in NEEDED_EVENT:
                    try:
                        node = Node.create_from(nsys_json, datas[0]['data'])
                    except Exception as e:
                        print(
                            f"Create Node Failed\n    Json: {nsys_json}\n    Error: {e}\n"
                        )
                        raise e
                    if node.skip:
                        continue
                    if isinstance(node, CudaNode):
                        cudas.append(node)
                    else:
                        cpus.append(node)

    return cpus, cudas, datas


def filter_nodes(nodes, node_filter):
    start_idx = None
    stop_idx = None

    for idx, node in enumerate(nodes):
        res = node_filter(node)
        if res == FilterResult.SAVE and start_idx is None:
            start_idx = idx

        if res == FilterResult.BREAK:
            stop_idx = idx
            return nodes[start_idx:stop_idx]
    if start_idx is not None:
        return nodes[start_idx:]
    else:
        return []


def fill_tree_nodes(nodes, stacks, trees):
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


def create_tree(json_path, *filters):
    # 1. read file and create nodes
    cpus, cudas, datas = create_nodes(json_path)
    data = datas[0]['data']
    cpu_nodes = cpus

    # 2. filter cpu nodes
    for _filter in filters:
        cpu_nodes = filter_nodes(cpu_nodes, _filter)
    assert len(cpu_nodes) > 0

    # 3. build tree structure
    stacks = ddict(lambda: [])
    trees = ddict(lambda: [])
    fill_tree_nodes(cpu_nodes, stacks, trees)

    # 4. set correlation nodes
    correlationId_map = ddict(lambda: [])
    for node in cudas:
        correlationId_map[node.correlationId].append(node)

    cuda_api_nodes = []
    cuda_kernel_nodes = []

    for node in cpu_nodes:
        if isinstance(node, TraceProcessNode):
            if correlationId_map.has(node.correlationId):
                related = correlationId_map[node.correlationId][0]
                # correlationId_map[node.correlationId].pop(0)
                node.related = related
                related.related = node
                if related.tag == "kernel":
                    related.text = data[related.name]
                cuda_api_nodes.append(node)
                cuda_kernel_nodes.append(related)
            else:
                node.related = None

    return Tree(
        trees=trees,
        datas=datas,
        cpu_nodes=cpu_nodes,
        cuda_api_nodes=cuda_api_nodes,
        cuda_kernel_nodes=cuda_kernel_nodes,
        main_thread=cpu_nodes[0].thread,
    )
