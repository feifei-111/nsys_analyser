from __future__ import annotations

import json
from collections import defaultdict

from ..nodes import CpuNode, CudaNode, TraceProcessNode, NvtxNode

NEEDED_TABLES = {
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "NVTX_EVENTS",
}

CUDA_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
}


def _extract_pid(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def _create_cuda_node(obj: dict, table: str, data: dict) -> CudaNode | None:
    kwargs = dict(
        start=obj["start"],
        end=obj["end"],
        event_name="CudaEvent",
        correlationId=obj["correlationId"],
        deviceID=obj["deviceId"],
        process_id=_extract_pid(obj["globalPid"]),
    )

    if table == "CUPTI_ACTIVITY_KIND_KERNEL":
        kwargs.update(
            tag="kernel", text=data[obj["shortName"]], cuda_action=obj
        )
    elif table == "CUPTI_ACTIVITY_KIND_MEMCPY":
        kwargs.update(tag="memcpy", cuda_action=obj)
    elif table == "CUPTI_ACTIVITY_KIND_MEMSET":
        kwargs.update(tag="memset", cuda_action=obj)
    elif table == "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION":
        kwargs.update(tag="sync", cuda_action=obj)
    else:
        return None

    return CudaNode(**kwargs)


def _create_runtime_node(obj: dict, data: dict) -> TraceProcessNode:
    return TraceProcessNode(
        start=obj["start"],
        end=obj["end"],
        event_name="TraceProcessEvent",
        correlationId=obj["correlationId"],
        thread=obj["globalTid"] & 0xFFFFFF,
        text=data[obj["nameId"]],
        event_class=obj.get("eventClass", 0),
        process_id=_extract_pid(obj["globalTid"]),
    )


def _create_nvtx_node(obj: dict, data: dict) -> NvtxNode | None:
    et = obj.get("eventType")
    if et == 59:
        textid = obj.get("textId", -1)
        text = data.get(textid, "") if textid != -1 else ""
        if "end" not in obj:
            return None
        return NvtxNode(
            start=obj["start"],
            end=obj["end"],
            event_name="NvtxEvent",
            thread=obj["globalTid"] & 0xFFFFFF,
            text=text,
            domain=obj.get("domainId", 0),
        )
    elif "end" in obj and "text" in obj:
        return NvtxNode(
            start=obj["start"],
            end=obj["end"],
            event_name="NvtxEvent",
            thread=obj["globalTid"] & 0xFFFFFF,
            text=obj["text"],
            domain=obj.get("domainId", 0),
        )
    else:
        return None


def create_nodes(
    json_path: str,
) -> tuple[
    dict[int, dict[int, list[CpuNode]]],  # {process_id: {thread_id: [nodes]}}
    dict[int, list[CudaNode]],  # {process_id: [cuda_nodes]}
    dict[int, str],  # string data
    dict,  # meta
]:
    data: dict[int, str] = {}
    cpu_groups: dict[int, dict[int, list[CpuNode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cuda_groups: dict[int, list[CudaNode]] = defaultdict(list)
    processes: dict[int, dict] = {}
    thread_names_raw: list[tuple[int, int]] = []

    with open(json_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            table = obj.get("table", "")

            if table == "StringIds":
                data[obj["id"]] = obj["value"]
                continue

            if table == "PROCESSES":
                processes[obj["pid"]] = obj["name"]
                continue

            if table == "ThreadNames":
                thread_names_raw.append((obj["globalTid"], obj["nameId"]))
                continue

            if table not in NEEDED_TABLES:
                continue

            try:
                if table in CUDA_TABLES:
                    node = _create_cuda_node(obj, table, data)
                elif table == "CUPTI_ACTIVITY_KIND_RUNTIME":
                    node = _create_runtime_node(obj, data)
                elif table == "NVTX_EVENTS":
                    node = _create_nvtx_node(obj, data)
                else:
                    continue
            except Exception as e:
                print(
                    f"Create Node Failed\n    Table: {table}\n    Json: {obj}\n    Error: {e}\n"
                )
                raise

            if node is None or node.skip:
                continue
            if isinstance(node, CudaNode):
                cuda_groups[node.process_id].append(node)
            else:
                pid = _extract_pid(obj["globalTid"])
                cpu_groups[pid][node.thread].append(node)

    # sort each thread's nodes by start time
    for threads in cpu_groups.values():
        for tid in threads:
            threads[tid].sort(key=lambda n: n.start)
    for pid in cuda_groups:
        cuda_groups[pid].sort(key=lambda n: n.start)

    thread_names = {
        tid & 0xFFFFFF: data.get(nid, f"id:{nid}")
        for tid, nid in thread_names_raw
    }
    meta = {"processes": processes, "thread_names": thread_names}

    return dict(cpu_groups), dict(cuda_groups), data, meta
