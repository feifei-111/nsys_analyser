from .utils import DefaultDict as ddict
from .utils import line_printer, LINE_WIDTH, sort_on_values


def analyse_ops_with_multi_thread(tree):
    op_time_cost = {}
    op_counter = {}
    for op_name in tree.op_set:

        start_times = [op.start for op in tree.nodes if op.is_op and op.op_name == op_name]
        end_times = [op.end for op in tree.nodes if op.is_op and op.op_name == op_name]

        start_time_idx = 0
        end_time_idx = 0

        last_start = None
        counter = 0

        total_time_cost = 0

        while start_time_idx < len(start_times) or end_time_idx < len(end_times):
            start_t = start_times[start_time_idx] if start_time_idx < len(start_times) else float("inf")
            end_t = end_times[end_time_idx] if end_time_idx < len(end_times) else float("inf")

            if start_t < end_t:
                if counter == 0:
                    last_start = start_t
                counter += 1
                start_time_idx += 1
            else:
                counter -= 1
                if counter == 0:
                    time_cost = end_t - last_start
                    total_time_cost += time_cost
                    last_start = None
                end_time_idx += 1

        op_time_cost[op_name] = total_time_cost
        op_counter[op_name] = len(start_times)

    print("{k:<40s}: time_cost = {v:<12f} ms".format(k="total", v=tree.time_cost / 1000000))
    print("-" * LINE_WIDTH)
    for k, v in sort_on_values(op_time_cost):
        print("{k:<40s}: count = {count:<5d}, time_cost = {v:<12f} ms,   {ratio:.2f}%".format(count=op_counter[k], k=k, v=v / 1000000, ratio=v / tree.time_cost * 100))


def show_op_list(tree):
    for node in tree.nodes:
        if node.is_op:
            print("{text:<40s}: time_cost = {time_cost:<8d} ms,  start = {start:<12d},  end = {end:<12d}".format(
                text=node.text, time_cost=node.time_cost, start=node.start, end=node.end
            ))
