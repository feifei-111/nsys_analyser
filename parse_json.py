import json

class Node:
    def __init__(self, nvtx_event):
        start = nvtx_event.get("Timestamp", None)
        end = nvtx_event.get("EndTimestamp", None)
        text = nvtx_event.get("Text", None)
        self.start = int(start) if start is not None else None
        self.end = int(end) if end is not None else None
        self.text = text
        self.children = []

    @property
    def time_cost(self):
        return self.end - self.start

    def __repr__(self):
        return f"text: {self.text}, start: {self.start}, end: {self.end}"
    
    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if level != 0:
            for child in self.children:
                child.pprint(level=level-1, prefix=prefix + "    ")

    def under(self, father):
        return father.start < self.start and father.end > self.end

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


def create_tree(nodes, target_step):
    def fill_tree_nodes(nodes, stack):
        for node in nodes:

            while not node.under(stack[-1]):
                if len(stack) == 1:
                    return stack[-1]
                stack.pop()

            stack[-1].children.append(node)
            stack.append(node)

    for idx, node in enumerate(nodes):
        if node.text == target_step:
            return fill_tree_nodes(nodes[idx + 1:], [node])