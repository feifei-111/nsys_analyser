import json
from .utils import DefaultDict as ddict

class Node:
    def __init__(self, nvtx_event):
        start = nvtx_event.get("Timestamp", None)
        end = nvtx_event.get("EndTimestamp", None)
        text = nvtx_event.get("Text", None)
        thread = nvtx_event["GlobalTid"]
        self.start = int(start) if start is not None else None
        self.end = int(end) if end is not None else None
        self.text = text
        self.children = []
        self.thread = thread
        self.parent = None

    @property
    def time_cost(self):
        return self.end - self.start

    def __repr__(self):
        return "text: {text:<40s},  cost: {cost:<10d}, start: {start:<12d}, end: {end:<12d}".format(text=self.text, start=self.start, end=self.end, cost=self.time_cost)
    
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

    def is_static_op(self):
        return len(self.find_child("compute")) == 1 and len(self.find_child("infer_shape")) == 1
    
    def is_dynamic_op(self):
        return "dygraph" in self.text or "pybind_imperative_func" in self.text or "pybind_patch_func" in self.text

    def is_other_op(self):
        return self.text in (
            "StreamSafeCUDAAllocator::Free", 
            "BufferedReader:MemoryCopy"
        )

    def can_be_op(self):
        if self.parent is None:
            return True
        if self.parent.can_be_op():
            if self.parent.is_op():
                return False
            else:
                return True
        else:
            return False

    def is_op(self):
        if self.can_be_op():
            return self.is_static_op() or self.is_dynamic_op() or self.is_other_op()
        else:
            return False


class Tree:
    def __init__(self, trees, **kwargs):
        self.trees = trees
        self.main_thread = kwargs.get("main_thread", None)
        self.nodes = kwargs.get("nodes", None)

        self.main_root = self.trees[self.main_thread]
        self.start = self.trees[self.main_thread][0].start
        self.end = self.trees[self.main_thread][-1].end
        self.op_set = set(node.text for node in self.nodes if node.is_op())

    def traversal_all(self):
        def inner():
            yield self.main_root
            for k,v in self.trees.items():
                if k != self.main_thread:
                    yield v
        return inner()

    def traversal_sub_threads(self):
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
