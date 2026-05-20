from __future__ import annotations

import functools


class Node:
    def __init__(self, *, start, end, event_name, skip=False):
        self.start = int(start)
        self.end = int(end)
        self.event_name = event_name
        self.skip = skip

    @property
    def time_cost(self):
        return self.end - self.start

    def time_under(self, father):
        return father.start < self.start and father.end > self.end

    def __repr__(self):
        return f"[{self.event_name}]  {self.to_string()}"


class CudaNode(Node):
    def __init__(
        self,
        *,
        tag,
        correlationId,
        deviceID,
        text=None,
        cuda_action=None,
        process_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tag = tag
        self.process_id = process_id
        self.correlationId = correlationId
        self.deviceID = int(deviceID)
        self.text = text if text is not None else tag
        self.cuda_action = cuda_action or {}
        self.related = None

        if self.correlationId == 0:
            self.skip = True

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(
            text=self.text,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
        )


class CpuNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.related: list[CudaNode] = []

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
    def kernels(self) -> dict[int, list]:
        """Returns {device_id: [CudaNode, ...]}."""
        result: dict[int, list] = {}
        if self.related:
            for r in self.related:
                result.setdefault(r.deviceID, []).append(r)
        else:
            for child in self.children:
                for dev, nodes in child.kernels().items():
                    result.setdefault(dev, []).extend(nodes)
        return result

    @functools.lru_cache
    def kernel_cover_time(self) -> dict[int, tuple[int, int]]:
        """Returns {device_id: (cover_time, range_time)}."""
        return {
            dev: nodes_time_cover(nodes)
            for dev, nodes in self.kernels().items()
        }

    def pprint(self, level=-1, prefix=""):
        print(prefix + self.__repr__())
        if isinstance(self, NvtxNode) and level != 0:
            for child in self.children:
                child.pprint(level=level - 1, prefix=prefix + "    ")


class TraceProcessNode(CpuNode):
    def __init__(
        self,
        *,
        correlationId,
        thread,
        text,
        event_class=0,
        process_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.process_id = process_id
        self.correlationId = correlationId
        self.thread = thread
        self.text = text
        self.event_class = int(event_class)
        self.children = []
        self.parent = None

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
    def __init__(self, *, thread, text, domain=0, **kwargs):
        super().__init__(**kwargs)
        self.thread = thread
        self.text = text
        self.domain = domain
        self.children = []
        self.parent = None

        if not self.text:
            self.skip = True

    def to_string(self):
        return "{text:<35s}:  time_cost = {cost:<8s} us,  start = {start:<10d},  end = {end:<10d}".format(
            text=self.text,
            start=self.start,
            end=self.end,
            cost=str(self.time_cost / 1000),
        )


# this function is for calc event time which might overlap with each other
def nodes_time_cover(nodes):
    if not nodes:
        return 0, 0

    start_times = [node.start for node in nodes]
    end_times = [node.end for node in nodes]

    start_times.sort()
    end_times.sort()

    range_time = end_times[-1] - start_times[0]

    start_time_idx = 0
    end_time_idx = 0

    last_start = None
    counter = 0

    total_time_cost = 0

    # we get the small time from the head of start_times or end_times
    # the idx show where the head is

    # if still have time not used
    while start_time_idx < len(start_times) or end_time_idx < len(end_times):
        next_start_time = (
            start_times[start_time_idx]
            if start_time_idx < len(start_times)
            else float("inf")
        )
        next_end_time = (
            end_times[end_time_idx]
            if end_time_idx < len(end_times)
            else float("inf")
        )

        if next_start_time < next_end_time:
            if counter == 0:
                last_start = next_start_time
            counter += 1
            start_time_idx += 1
        else:
            counter -= 1
            if counter == 0:
                time_cost = next_end_time - last_start
                total_time_cost += time_cost
                last_start = None
            end_time_idx += 1

    return total_time_cost, range_time
