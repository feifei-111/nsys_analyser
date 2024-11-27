import functools
from .node_status import nodes_time_cover

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

    @property
    def time_cost(self):
        return self.end - self.start

    def time_under(self, father):
        return father.start < self.start and father.end > self.end

    def __repr__(self):
        return f"[{self.event_name}]  {self.to_string()}"


class CudaNode(Node):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
        cuda_event = event_json["CudaEvent"]
        self.start = int(cuda_event["startNs"])
        self.end = int(cuda_event["endNs"])
        self.text = None
        self.correlationId = cuda_event["correlationId"]
        self.deviceID = int(cuda_event["deviceId"])
        self.event_class = int(cuda_event["eventClass"])
        self.related = None

        if self.type == 79:
            self.tag = "kernel"
            self.name = int(cuda_event["kernel"]["shortName"])
            self.text = data[self.name]
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

        if self.text is None:
            self.text = self.tag

        self.cuda_action = cuda_event[self.tag]
        if self.correlationId == 0:
            self.skip = True

        if self.event_class != 0:
            # 1: memcpy
            # 2: memset
            # 3: kernel
            # 4: (unknow)
            # 5: sync
            pass

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(
            text=self.text,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
        )


class CpuNode(Node):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
        self.related = None
        self._kernel_cover_time = None
        self._kernel_range_time = None

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

    def find_surface(self, checker):
        if isinstance(checker, str):
            return self.find_surface(lambda x: x.text == checker)
        result = []
        if checker(self):
            result.append(self)
        else:
            for child in self.children:
                result.extend(child.find_surface(checker))
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

    @functools.lru_cache
    def kernels(self):
        if self.related is not None:
            return [self.related]
        else:
            kernels = []
            for child in self.children:
                kernels.extend(child.kernels())
            return kernels

    def kernel_cover_time(self):
        if self._kernel_cover_time is None:
            self._kernel_cover_time, self._kernel_range_time = nodes_time_cover(
                self.kernels()
            )
        return self._kernel_cover_time

    def kernel_ranged_time(self):
        if self._kernel_range_time is None:
            self._kernel_cover_time, self._kernel_range_time = nodes_time_cover(
                self.kernels()
            )
        return self._kernel_range_time

    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if isinstance(self, NvtxNode) and level != 0:
            for child in self.children:
                child.pprint(level=level - 1, prefix=prefix + "    ")


class TraceProcessNode(CpuNode):
    def __init__(self, event_json, data):
        super().__init__(event_json, data)
        trace_event = event_json["TraceProcessEvent"]
        self.correlationId = trace_event["correlationId"]
        self.thread = trace_event["globalTid"]
        self.children = []
        self.parent = None
        self.name = int(trace_event["name"])
        self.text = data[self.name]
        self.start = int(trace_event["startNs"])
        self.end = int(trace_event["endNs"])
        self.event_class = int(trace_event["eventClass"])

        if self.event_class != 0:
            # epoll, TLS, futex
            pass

        if self.correlationId == 0:
            self.skip = True

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}, correlationId = {correlationId:<10d}".format(
            text=self.text,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
            correlationId=self.correlationId,
        )


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


NAME_NODE_MAP = {
    "NvtxEvent": NvtxNode,
    "TraceProcessEvent": TraceProcessNode,
    "CudaEvent": CudaNode,
}


def create_node(event_json, data):
    event_name = EVENT_TYPE_NAME[event_json['Type']]
    return NAME_NODE_MAP[event_name](event_json, data)
