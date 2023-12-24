EVENT_TYPE_NAME ={
    27: 'CommEvent',
    47: 'TraceProcessEvent',
    48: 'TraceProcessEvent',    # call cuda api in cpu
    49: 'DiagnosticEvent',
    59: 'NvtxEvent',
    79: 'CudaEvent',            # kernel
    80: 'CudaEvent',            # memcpy/memset
    106: 'CudaEvent',           # sync
    127: 'CudaEvent'            # cudaEventRecord
}

NEEDED_EVENT = {
    48, 59, 79, 80, 106
}


class Node:
    def __init__(self, event_json):
        self.json = event_json
        self.type = event_json['Type']
        self.event_name = EVENT_TYPE_NAME[self.type]

    @staticmethod
    def create_from(event_json):
        name_node_map = {
            "NvtxEvent": NvtxNode,
            "TraceProcessEvent": TraceProcessNode,
            "CudaEvent": CudaNode,
        }
        event_name = EVENT_TYPE_NAME[event_json['Type']]
        return name_node_map[event_name](event_json)

    @property
    def time_cost(self):
        return self.end - self.start

    def __repr__(self):
        return f"[{self.event_name}]  {self.to_string()}"

    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if isinstance(self, NvtxNode) and level != 0:
            for child in self.children:
                child.pprint(level=level-1, prefix=prefix + "    ")


class CudaNode(Node):
    def __init__(self, event_json):
        super().__init__(event_json)
        cuda_event = event_json["CudaEvent"]
        self.start = int(cuda_event["startNs"])
        self.end = int(cuda_event["endNs"])
        self.text = None
        self.correlationId = cuda_event["correlationId"]
        self.event_class = cuda_event["eventClass"]
        self.related = None

        if self.type == 79:
            self.tag = "kernel"
            self.name = int(cuda_event["kernel"]["shortName"])
        elif self.type == 80:
            if "memcpy" in cuda_event:
                self.tag = "memcpy"
            elif "memset" in cuda_event:
                self.tag = "memset"
            else:
                raise RuntimeError("Unknown cuda event")
        elif self.type == 106:
            self.tag = "sync"
        else:
            raise RuntimeError("Unknown cuda event")

        self.cuda_action = cuda_event[self.tag]
        if self.correlationId == 0:
            self.type = -1
    
    @property
    def kernel_name(self):
        return self.text if self.text is not None else self.tag

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(text=self.kernel_name, start=self.start, end=self.end, cost=str(self.time_cost/1000))


class CpuNode(Node):
    def __init__(self, event_json):
        super().__init__(event_json)
        self.is_op = False
        self._kernels = None

    def find_child(self, checker):
        if isinstance(checker, str):
            return self.find_child(lambda x: x.text == checker)
        result = []
        for child in self.children:
            if checker(child):
                result.append(child)
        return result

    def find_all(self, checker):
        if isinstance(checker, str):
            return self.find_all(lambda x: x.text == checker)
        result = []
        if checker(self):
            result.append(self)
        for child in self.children:
            result.extend(child.find_all(checker))
        return result

    def has(self, checker):
        if isinstance(checker, str):
            return self.has(lambda x: x.text == checker)
        if checker(self):
            return True
        return any(child.has(checker) for child in self.children)

    def traversal(self):
        yield self
        for child in self.children:
            yield from child.traversal()

    def time_under(self, father):
        return father.start < self.start and father.end > self.end

    def under(self, father):
        return father.thread == self.thread and self.time_under(father)
    
    def set_op(self):
        return set()

    def kernels(self):
        if self._kernels is None:
            self._kernels = [x.related for x in self.find_all(lambda x: isinstance(x, TraceProcessNode)) if x.related is not None]
        return self._kernels

    def kernel_time(self):
        return sum(x.time_cost for x in self.kernels())


class TraceProcessNode(CpuNode):
    def __init__(self, event_json):
        super().__init__(event_json)
        trace_event = event_json["TraceProcessEvent"]
        self.start = int(trace_event["startNs"])
        self.end = int(trace_event["endNs"])
        self.correlationId = trace_event["correlationId"]
        self.thread = trace_event["globalTid"]
        self.text = None
        self.children = tuple()
        self.parent = None
        self.related = None
        self.name = int(trace_event["name"])
        
        if self.correlationId == 0:
            self.type = -1

    def to_string(self):
        if self.related is None:
            return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}, correlationId = {correlationId:<10d}".format(text=self.text, start=self.start, end=self.end, cost=str(self.time_cost/1000), correlationId=self.correlationId)
        else:
            return "=>  " + str(self.related)


class NvtxNode(CpuNode):
    def __init__(self, event_json):
        super().__init__(event_json)
        nvtx_event = event_json["NvtxEvent"]
        self.start = int(nvtx_event["Timestamp"])
        self.end = int(nvtx_event["EndTimestamp"])
        self.thread = nvtx_event["GlobalTid"]
        self.text = nvtx_event.get("Text", "")
        self.children = []
        self.parent = None

        if self.text == "":
            self.type = -1

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(text=self.text, start=self.start, end=self.end, cost=str(self.time_cost/1000))

    def set_op(self):
        ret = set()
        if self._maybe_op():
            ret.add(self._mark_as_op())
        else:
            for child in self.children:
                ret.update(child.set_op())
        return ret

    def _maybe_op(self):
        def is_other_op(text):
            other_ops = ("StreamSafeCUDAAllocator::Free", "BufferedReader:MemoryCopy")
            other_op_prefix = ("GpuMemcpyAsync:")
            return text in other_ops or any(text.startswith(prefix) for prefix in other_op_prefix)

        return (
            (len(self.find_child("compute")) == 1 and len(self.find_child("infer_shape")) == 1) or
            ("pd_op." in self.text) or
            ("dygraph" in self.text or "pybind_imperative_func" in self.text or "pybind_patch_func" in self.text) or
            (is_other_op(self.text))
        )

    def _mark_as_op(self):
        self.is_op = True
        if "dygraph" in self.text or "pybind_imperative_func" in self.text or "pybind_patch_func" in self.text:
            self.op_name = "[D]  " + self.text.replace(" dygraph", "").replace(" pybind_imperative_func", "").replace(" pybind_patch_func", "")
        elif "pd_op." in self.text:
            self.op_name = self.text.replace("pd_op.", "")
        else:
            self.op_name = self.text
        return self.op_name


class Tree:
    def __init__(self, **kwargs):
        self.trees = kwargs["trees"]
        self.datas = kwargs["datas"]
        self.nodes = kwargs["nodes"]
        self.op_set = kwargs["op_set"]
        self.main_thread = kwargs["main_thread"]

        self.main_roots = self.trees[self.main_thread]
        self.start = self.trees[self.main_thread][0].start
        self.end = self.trees[self.main_thread][-1].end

    @property
    def time_cost(self):
        return self.end - self.start

    def all_roots(self):
        for k, roots in self.trees.items():
            for root in roots:
                yield root

