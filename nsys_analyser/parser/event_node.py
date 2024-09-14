EVENT_TYPE_NAME = {
    27: 'CommEvent',
    47: 'TraceProcessEvent',
    48: 'TraceProcessEvent',  # call cuda api in cpu
    49: 'DiagnosticEvent',
    59: 'NvtxEvent',  # if NvtxEvent is 59, it has "TextId"
    60: 'NvtxEvent',  # if NvtxEvent is 60, it has "Text"
    79: 'CudaEvent',  # kernel
    80: 'CudaEvent',  # memcpy/memset
    106: 'CudaEvent',  # sync
    127: 'CudaEvent',  # cudaEventRecord
    31: 'CompositeEvent',  # call libxx.so api, e.g. libcublasLt.so, libc-2.31.so, looks like a stack
}

NEEDED_EVENT = {48, 59, 79, 80, 106}


class Node:
    def __init__(self, event_json, data):
        self.json = event_json
        self.type = event_json['Type']
        self.event_name = EVENT_TYPE_NAME[self.type]
        self.skip = False
        self.start = None
        self.end = None

    @staticmethod
    def create_from(event_json, data):
        name_node_map = {
            "NvtxEvent": NvtxNode,
            "TraceProcessEvent": TraceProcessNode,
            "CudaEvent": CudaNode,
        }
        event_name = EVENT_TYPE_NAME[event_json['Type']]
        return name_node_map[event_name](event_json, data)

    @property
    def time_cost(self):
        return self.end - self.start

    def time_under(self, father):
        return father.start < self.start and father.end > self.end

    def __repr__(self):
        return f"[{self.event_name}]  {self.to_string()}"

    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if isinstance(self, NvtxNode) and level != 0:
            for child in self.children:
                child.pprint(level=level - 1, prefix=prefix + "    ")


class CudaNode(Node):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
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
            self.skip = True

    @property
    def kernel_name(self):
        return self.text if self.text is not None else self.tag

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(
            text=self.kernel_name,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
        )


class CpuNode(Node):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
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

    def under(self, father):
        return father.thread == self.thread and self.time_under(father)

    def kernels(self):
        if self._kernels is None:
            self._kernels = [
                x.related
                for x in self.find_all(
                    lambda x: isinstance(x, TraceProcessNode)
                )
                if x.related is not None
            ]
        return self._kernels

    def kernel_time(self):
        return sum(x.time_cost for x in self.kernels())


class TraceProcessNode(CpuNode):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
        trace_event = event_json["TraceProcessEvent"]
        self.correlationId = trace_event["correlationId"]
        self.thread = trace_event["globalTid"]
        self.children = []
        self.parent = None
        self.related = None
        self.name = int(trace_event["name"])
        self.text = data[self.name]
        if self.correlationId == 0:
            self.start = 0
            self.skip = False
        else:
            self.start = int(trace_event["startNs"])
        self.end = int(trace_event["endNs"])

    def to_string(self):
        if self.related is None:
            return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}, correlationId = {correlationId:<10d}".format(
                text=self.text,
                start=self.start,
                end=self.end,
                cost=str(self.time_cost / 1000),
                correlationId=self.correlationId,
            )
        else:
            return "=>  " + str(self.related)


class NvtxNode(CpuNode):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
        nvtx_event = event_json["NvtxEvent"]
        self.start = int(nvtx_event["Timestamp"])
        self.end = int(nvtx_event["EndTimestamp"])
        self.thread = nvtx_event["GlobalTid"]
        if self.type == 59:
            self.textid = int(nvtx_event.get("TextId", -1))
            if self.textid == -1:
                self.text = ""
                self.skip = True
            else:
                self.text = data[self.textid]
        elif self.type == 60:
            self.text = nvtx_event.get("Text", "")
            if self.text == "":
                self.skip = True
        self.children = []
        self.parent = None

        self.domain = nvtx_event[
            "DomainId"
        ]  # different domain will show block in different row in nsight graph, e.g. DomainId=1 means TensorRT

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(
            text=self.text,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
        )
