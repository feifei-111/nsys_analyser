from ..report import LINE_WIDTH, ReportTitle, log
from ..utils import DefaultDict as ddict
from ..utils import sort_on_values
from .node_status import nodes_time_cover


def analyse_kernel_ratio(tree):
    total_time = 0
    total_kernel_time = 0
    for root in tree.main_roots:
        cover_time = nodes_time_cover(
            [node for node in tree.cuda_kernel_nodes if node.time_under(root)]
        )
        total_kernel_time += cover_time
        total_time += root.time_cost

    with ReportTitle("kernel_ratio"):
        log(
            "total_time: {total_time:<10f} ms, total_kernel_time: {total_kernel_time:<10f}, kernel_ratio: {ratio:<.2f}".format(
                total_time=total_time / 1000000,
                total_kernel_time=total_kernel_time / 1000000,
                ratio=total_kernel_time / total_time * 100,
            )
        )


def analyse_kernel_time_cost(tree):
    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    total_time = 0
    for kernel in tree.cuda_kernel_nodes:
        kernel_time_cost[kernel.kernel_name] += kernel.time_cost
        total_time += kernel.time_cost
        kernel_count[kernel.kernel_name] += 1

    with ReportTitle("kernel_time_cost"):
        log(
            "{k:<40s}:  time_cost = {v:<10f} ms".format(
                k="total_kernel_time", v=total_time / 1000000
            )
        )
        log("-" * LINE_WIDTH)
        for k, v in sort_on_values(kernel_time_cost):
            log(
                "{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms,  count = {kernel_count:<5d}".format(
                    k=k, kernel_cost=v / 1000000, kernel_count=kernel_count[k]
                )
            )


def analyse_kernel_list(tree):
    with ReportTitle("kernel_list"):
        for kernel in tree.cuda_kernel_nodes:
            log(str(kernel))
