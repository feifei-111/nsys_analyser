from __future__ import annotations

from typing import Callable

from .tree import CpuGroups


def _filter_time_range_ns(
    cpu_groups: CpuGroups, start_ns: int, end_ns: int
) -> CpuGroups:
    new_groups: CpuGroups = {}
    for pid, threads in cpu_groups.items():
        new_threads = {}
        for tid, nodes in threads.items():
            filtered = []
            for n in nodes:
                if n.start < start_ns:
                    continue
                if n.start >= end_ns:
                    break
                filtered.append(n)
            if filtered:
                new_threads[tid] = filtered
        if new_threads:
            new_groups[pid] = new_threads
    return new_groups


class NodeFilter:

    def __init__(self):
        self._filters = []

    def __call__(self, cpu_groups: CpuGroups) -> CpuGroups:
        for filter in self._filters:
            cpu_groups = filter(cpu_groups)
        return cpu_groups

    def time_range(self, start: float, end: float):
        """Keep events within [start, end) time range. Input in seconds."""
        start_ns = int(start * 1e9)
        end_ns = int(end * 1e9)
        self._filters.append(
            lambda g: _filter_time_range_ns(g, start_ns, end_ns)
        )
        return self

    def first_text(self, text: str):
        """Keep events under the first event matching text."""

        def _filter(cpu_groups: CpuGroups) -> CpuGroups:
            first = None
            for threads in cpu_groups.values():
                for nodes in threads.values():
                    for n in nodes:
                        if getattr(n, 'text', None) == text:
                            if first is None or n.start < first.start:
                                first = n
                            break
            if first is None:
                return {}
            return _filter_time_range_ns(cpu_groups, first.start, first.end)

        self._filters.append(_filter)
        return self

    def all_text(self, text: str):
        """Keep events under all events matching text."""

        def _filter(cpu_groups: CpuGroups) -> CpuGroups:
            ranges = []
            for threads in cpu_groups.values():
                for nodes in threads.values():
                    for n in nodes:
                        if getattr(n, 'text', None) == text:
                            ranges.append((n.start, n.end))
            if not ranges:
                return {}
            ranges.sort()
            new_groups: CpuGroups = {}
            for pid, threads in cpu_groups.items():
                new_threads = {}
                for tid, nodes in threads.items():
                    filtered = [
                        n
                        for n in nodes
                        if any(s <= n.start < e for s, e in ranges)
                    ]
                    if filtered:
                        new_threads[tid] = filtered
                if new_threads:
                    new_groups[pid] = new_threads
            return new_groups

        self._filters.append(_filter)
        return self

    def process(self, *pids: int):
        """Keep threads belonging to given process ids (outer keys of cpu_groups)."""
        keep_pids = set(pids)

        def _filter(cpu_groups: CpuGroups) -> CpuGroups:
            return {
                pid: threads
                for pid, threads in cpu_groups.items()
                if pid in keep_pids
            }

        self._filters.append(_filter)
        return self

    def thread(self, *tids: int):
        """Keep only given threads (inner keys of cpu_groups)."""
        keep_tids = set(tids)

        def _filter(cpu_groups: CpuGroups) -> CpuGroups:
            new_groups: CpuGroups = {}
            for pid, threads in cpu_groups.items():
                filtered = {
                    tid: nodes
                    for tid, nodes in threads.items()
                    if tid in keep_tids
                }
                if filtered:
                    new_groups[pid] = filtered
            return new_groups

        self._filters.append(_filter)
        return self
