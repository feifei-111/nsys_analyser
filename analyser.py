from .utils import DefaultDict as ddict
from .utils import line_printer, LINE_WIDTH


def find_op_base(base_events):
    events = []
    for event in base_events:
        if event.has("interpreter_core_run"):
            events.extend(event.find_all("interpreter_core_run"))
        else:
            events.append(event)
    return events


def analyse_ops_under_named_event(root, name):
    base_events = root.find_all(name)
    total_time = sum(base_event.time_cost for base_event in base_events)

    events = find_op_base(base_events)

    op_time_cost_map = ddict(0)
    op_counter = ddict(0)

    for event in events:
        for op in event.children:
            op_time_cost_map[op.text] += op.time_cost
            op_counter[op.text] += 1

    print("{k:<40s}: time_cost = {v:<12f} ms,   {ratio:.2f}%".format(k="total", v=total_time / 1000000, ratio=100))
    print("-" * LINE_WIDTH)
    for k, v in op_time_cost_map.sort_on_values():
        print("{k:<40s}: count = {count:<5d}, time_cost = {v:<12f} ms,   {ratio:.2f}%".format(count=op_counter[k], k=k, v=v / 1000000, ratio=v / total_time * 100))


def show_op_list(root, name):
    base_events = root.find_all(name)
    events = find_op_base(base_events)
    first_one = True

    for event in events:
        event.pprint(level=1)
        if not first_one:
            print("~" * LINE_WIDTH + "\n")
        else:
            first_one = False