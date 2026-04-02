from __future__ import annotations

import json
from collections import defaultdict

from ..nodes import CpuNode, CudaNode, TraceProcessNode, NvtxNode

EVENT_TYPE_NAME = {
    27: 'CommEvent',
    47: 'TraceProcessEvent',
    48: 'TraceProcessEvent',
    49: 'DiagnosticEvent',
    59: 'NvtxEvent',
    60: 'NvtxEvent',
    79: 'CudaEvent',
    80: 'CudaEvent',
    106: 'CudaEvent',
    127: 'CudaEvent',
    31: 'CompositeEvent',
}

NEEDED_EVENT = {48, 59, 79, 80, 106}


def _extract_pid(global_id: int) -> int:
    return global_id >> 24


def _create_cuda_node(event_json: dict, data) -> CudaNode:
    etype = event_json["Type"]
    ce = event_json["CudaEvent"]

    kwargs = dict(
        start=ce["startNs"],
        end=ce["endNs"],
        event_name="CudaEvent",
        correlationId=ce["correlationId"],
        deviceID=ce["deviceId"],
    )

    if etype == 79:
        name_idx = int(ce["kernel"]["shortName"])
        kwargs.update(
            tag="kernel", text=data[name_idx], cuda_action=ce.get("kernel", {})
        )
    elif etype == 80:
        if "memcpy" in ce:
            kwargs.update(tag="memcpy", cuda_action=ce["memcpy"])
        elif "memset" in ce:
            kwargs.update(tag="memset", cuda_action=ce["memset"])
        else:
            raise RuntimeError("Unknown cuda event")
    elif etype == 106:
        kwargs.update(tag="sync", cuda_action=ce.get("sync", {}))
    else:
        raise RuntimeError("Unknown cuda event")

    return CudaNode(**kwargs)


def _create_cpu_node(event_json: dict, data) -> CpuNode:
    etype = event_json["Type"]

    if etype == 48:
        te = event_json["TraceProcessEvent"]
        name_idx = int(te["name"])
        return TraceProcessNode(
            start=te["startNs"],
            end=te["endNs"],
            event_name="TraceProcessEvent",
            correlationId=te["correlationId"],
            thread=te["globalTid"],
            text=data[name_idx],
            event_class=te.get("eventClass", 0),
        )
    elif etype in (59, 60):
        ne = event_json["NvtxEvent"]
        if etype == 59:
            textid = int(ne.get("TextId", -1))
            text = data[textid] if textid != -1 else ""
        else:
            text = ne.get("Text", "")
        return NvtxNode(
            start=ne["Timestamp"],
            end=ne["EndTimestamp"],
            event_name="NvtxEvent",
            thread=ne["GlobalTid"],
            text=text,
            domain=ne.get("DomainId", 0),
        )
    else:
        raise RuntimeError(f"Unknown cpu event type: {etype}")


def create_nodes(
    json_path: str,
) -> tuple[
    dict[int, dict[int, list[CpuNode]]],  # {process_id: {thread_id: [nodes]}}
    dict[int, list[CudaNode]],  # {process_id: [cuda_nodes]}
    dict | list,  # string data
    dict,  # meta
]:
    datas: list = []
    cpu_groups: dict[int, dict[int, list[CpuNode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cuda_groups: dict[int, list[CudaNode]] = defaultdict(list)

    with open(json_path, "r") as f:
        for line in f:
            nsys_json = json.loads(line)

            if "Type" not in nsys_json:
                datas.append(nsys_json)
            else:
                event_type = nsys_json["Type"]
                if event_type not in NEEDED_EVENT:
                    continue
                try:
                    data = datas[0]["data"]
                    if event_type in (79, 80, 106):
                        node = _create_cuda_node(nsys_json, data)
                    else:
                        node = _create_cpu_node(nsys_json, data)
                except Exception as e:
                    print(
                        f"Create Node Failed\n    Json: {nsys_json}\n    Error: {e}\n"
                    )
                    raise
                if node.skip:
                    continue
                if isinstance(node, CudaNode):
                    pid = getattr(node, 'process_id', 0)
                    cuda_groups[pid].append(node)
                else:
                    pid = _extract_pid(node.thread)
                    cpu_groups[pid][node.thread].append(node)

    meta: dict = {"processes": {}, "thread_names": {}}
    return dict(cpu_groups), dict(cuda_groups), datas[0]["data"], meta
