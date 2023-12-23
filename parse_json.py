import json
from .utils import DefaultDict as ddict
from .tree import *

def create_nodes(json_path):
    datas = []
    cudas = []
    cpus = []
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            nsys_json = json.loads(line)
            if "Type" in nsys_json:
                event_type = nsys_json["Type"]
                if event_type in NEEDED_EVENT:
                    node = Node.create_from(nsys_json)
                    if node.type == -1:
                        continue
                    if isinstance(node, CudaNode):
                        cudas.append(node)
                    else:
                        cpus.append(node)
            else:
                datas.append(nsys_json)
    return cpus, cudas, datas 


def filter_nodes(nodes, filter_):
    stop_time = -float('inf')
    start_idx = None
    stop_idx = None
    main_thread = None

    for idx, node in enumerate(nodes):
        if filter_(node) and start_idx is None:
            start_idx = idx
            main_thread = node.thread
        
        if filter_(node) and node.end > stop_time:
            stop_time = node.end
            stop_idx = idx
        
        if node.start < stop_time:
            stop_idx = idx

    return nodes[start_idx :stop_idx]


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
    cpus, cudas, datas = create_nodes(json_path)
    data = datas[0]['data']

    # filter
    stacks = ddict(lambda: [])
    trees = ddict(lambda: [])
    cpu_nodes = cpus
    for idx, f in enumerate(filters):
        if isinstance(f, str):
            filter_ = lambda x: x.text == f
        else:
            filter_ = f
        cpu_nodes = filter_nodes(cpu_nodes, filter_)
        if idx == 0:
            main_thread = cpu_nodes[0].thread

    # build tree structure
    fill_tree_nodes(cpu_nodes, stacks, trees)

    # set ops
    op_name_set = set()
    for k, roots in trees.items():
        for root in roots:
            op_name_set.update(root.set_op())

    # set correlation nodes
    correlationId_map = ddict(lambda: [])
    for node in cudas:
        correlationId_map[node.correlationId].append(node)

    for node in cpu_nodes:
        if isinstance(node, TraceProcessNode):
            if correlationId_map.has(node.correlationId):
                related = correlationId_map[node.correlationId][0]
                correlationId_map[node.correlationId].pop(0)
                node.related = related
                related.related = node
                if related.tag == "kernel":
                    related.text = data[related.name]
            else:
                node.related = None
            node.text = data[node.name]
    return Tree(trees=trees, datas=datas, nodes=cpu_nodes, op_set=op_name_set, main_thread=main_thread)

