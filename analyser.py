from .utils import DefaultDict as ddict
from .utils import line_printer, LINE_WIDTH, sort_on_values
from .tree import TraceProcessNode


def analyse_op_time_cost(tree):
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

    print("{k:<40s}:  time_cost = {v:<10f} ms".format(k="total", v=tree.time_cost / 1000000))
    print("-" * LINE_WIDTH)
    for k, v in sort_on_values(op_time_cost):
        print("{k:<40s}:  time_cost = {v:<10f} ms,  count = {count:<5d},  {ratio:.2f}%".format(count=op_counter[k], k=k, v=v / 1000000, ratio=v / tree.time_cost * 100))



def analyse_op_kernel_time_cost(tree):
    op_kernel_time_cost = ddict(0)
    for op in tree.nodes:
        if op.is_op:
            op_kernel_time_cost[op.op_name] += op.kernel_time()
    for k, v in sort_on_values(op_kernel_time_cost):
        print("{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms".format(k=k, kernel_cost= v / 1000000))


def analyse_kernel_time_cost(tree):
    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    total_time = 0
    for node in tree.nodes:
        if isinstance(node, TraceProcessNode) and node.related is not None:
            kernel = node.related
            kernel_time_cost[kernel.kernel_name] += kernel.time_cost
            total_time += kernel.time_cost
            kernel_count[kernel.kernel_name] += 1
    print("{k:<40s}:  time_cost = {v:<10f} ms".format(k="total", v=total_time / 1000000))
    print("-" * LINE_WIDTH)
    for k, v in sort_on_values(kernel_time_cost):
        print("{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms,  count = {kernel_count:<5d}".format(k=k, kernel_cost= v / 1000000, kernel_count= kernel_count[k]))


def show_op_list(tree):
    for node in tree.nodes:
        if node.is_op:
            print(str(node))


def show_kernel_list(tree):
    for node in tree.nodes:
        if isinstance(node, TraceProcessNode) and node.related is not None:
            kernel = node.related
            print(str(kernel))


def analyse_interpreter_run(tree):
    main_roots = tree.main_roots
    for root in main_roots:
        interpreter_nodes = root.find_all("interpreter_core_run")
        iterpreter_time_cost = sum(node.time_cost for node in interpreter_nodes)
        print(f"Graph Number:           {len(interpreter_nodes)}")
        print(f"Interpreter Run Ratio:  {iterpreter_time_cost / tree.time_cost * 100:.2f}%")
    