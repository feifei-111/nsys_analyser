import json
from .utils import DefaultDict as ddict

class Node:
    def __init__(self, nvtx_event):
        start = nvtx_event.get("Timestamp", None)
        end = nvtx_event.get("EndTimestamp", None)
        text = nvtx_event.get("Text", "")
        thread = nvtx_event["GlobalTid"]
        self.start = int(start) if start is not None else None
        self.end = int(end) if end is not None else None
        self.text = text
        self.children = []
        self.thread = thread
        self.parent = None
        self.is_op = False

    @property
    def time_cost(self):
        return self.end - self.start

    def __repr__(self):
        return "{text:<40s}: time_cost = {cost:<10s} us,  start = {start:<10d},  end = {end:<10d}".format(text=self.text, start=self.start, end=self.end, cost=str(self.time_cost/1000))
    
    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if level != 0:
            for child in self.children:
                child.pprint(level=level-1, prefix=prefix + "    ")

    def time_under(self, father):
        return father.start < self.start and father.end > self.end

    def under(self, father):
        return father.thread == self.thread and self.time_under(father)

    def find_child(self, text):
        result = []
        for child in self.children:
            if child.text == text:
                result.append(child)
        return result

    def find_all(self, text):
        result = []
        if self.text == text:
            result.append(self)
        for child in self.children:
            result.extend(child.find_all(text))
        return result

    def has(self, text):
        if self.text == text:
            return True
        return any(child.has(text) for child in self.children)


class Tree:
    def __init__(self, trees, **kwargs):
        self.trees = trees
        self.main_thread = kwargs.get("main_thread", None)
        self.nodes = kwargs.get("nodes", None)

        self.main_roots = self.trees[self.main_thread]
        self.start = self.trees[self.main_thread][0].start
        self.end = self.trees[self.main_thread][-1].end
        for root in self.main_roots:
            setup_op(root)
        for roots in self.traversal_sub_threads_root_list():
            for root in roots:
                setup_op(root)

        self.op_set = set(node.op_name for node in self.nodes if node.is_op)

    def traversal_all_threads_root_list(self):
        def inner():
            yield self.main_roots
            for k,v in self.trees.items():
                if k != self.main_thread:
                    yield v
        return inner()

    def traversal_sub_threads_root_list(self):
        def inner():
            for k,v in self.trees.items():
                if k != self.main_thread:
                    yield v
        return inner()

    @property
    def time_cost(self):
        return self.end - self.start


def create_nodes(json_path):
    nodes = []
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            nsys_json = json.loads(line)
            if isinstance(nsys_json, dict) and "NvtxEvent" in nsys_json:
                nvtx_event = nsys_json.get("NvtxEvent", None)
                node = Node(nvtx_event)
                nodes.append(node)
    return nodes


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


def create_tree(nodes, target_step, filter=None):
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

    stacks = ddict(lambda: [])
    trees = ddict(lambda: [])

    target_nodes = filter_nodes(nodes, lambda x: x.text == target_step)
    main_thread = target_nodes[0].thread
    if filter is not None:
        target_nodes = filter_nodes(target_nodes, filter)

    fill_tree_nodes(target_nodes, stacks, trees)
    return Tree(trees, main_thread=main_thread, nodes=target_nodes)


def is_static_op(node):
    return len(node.find_child("compute")) == 1 and len(node.find_child("infer_shape")) == 1

def is_pir_op(node):
    return "pd_op." in node.text

def maybe_dynamic_op(node):
    return "dygraph" in node.text or "pybind_imperative_func" in node.text or "pybind_patch_func" in node.text

def is_other_op(node):
    return node.text in (
        "StreamSafeCUDAAllocator::Free", 
        "BufferedReader:MemoryCopy"
    )

def maybe_op(node):
    return is_static_op(node) or is_other_op(node) or maybe_dynamic_op(node) or is_pir_op(node)


def setup_op(root):
    if "grad" in root.text:
        return

    if maybe_op(root):
        root.is_op = True
        if maybe_dynamic_op(root):
            root.op_name = root.text.replace(" dygraph", "").replace(" pybind_imperative_func", "").replace(" pybind_patch_func", "") + "    [D]"
        elif is_pir_op(root):
            root.op_name = root.text.replace("pd_op.", "")
        else:
            root.op_name = root.text
    else:
        for child in root.children:
            setup_op(child)