from ..report import LINE_WIDTH, ReportTitle, log
from ..utils import DefaultDict as ddict
from ..utils import sort_on_values
from .node_status import nodes_time_cover


def analyse_kernel_ratio(tree):
    total_ranged_time = 0
    total_covered_time = 0
    kernel_count = 0
    for root in tree.main_roots:
        target = [
            node.related
            for node in root.traversal()
            if node.related is not None
        ]
        cover_time, ranged_time = nodes_time_cover(target)
        total_covered_time += cover_time
        total_ranged_time += ranged_time
        kernel_count += len(target)

    with ReportTitle("kernel_ratio"):
        log(
            "ranged_time: {ranged_time:<10f} ms, kernel_covered: {kernel_covered:<10f}, kernel_count: {kernel_count:<4d}, ratio: {ratio:<.2f}".format(
                ranged_time=total_ranged_time / 1000000,
                kernel_covered=total_covered_time / 1000000,
                ratio=total_covered_time / total_ranged_time * 100,
                kernel_count=kernel_count,
            )
        )


def analyse_kernel_time_cost(tree):
    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    total_time = 0
    for kernel in tree.cuda_kernel_nodes:
        kernel_time_cost[kernel.text] += kernel.time_cost
        total_time += kernel.time_cost
        kernel_count[kernel.text] += 1

    with ReportTitle("kernel_time_cost"):
        log(
            "{k:<40s} {v:<10f} ms".format(
                k="total_kernel_time", v=total_time / 1000000
            )
        )
        log("-" * LINE_WIDTH)
        for k, v in sort_on_values(kernel_time_cost):
            log(
                "{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms,  count = {kernel_count:<5d}, ratio = {ratio:<.2f}".format(
                    k=k,
                    kernel_cost=v / 1000000,
                    kernel_count=kernel_count[k],
                    ratio=v / total_time * 100,
                )
            )


def analyse_kernel_list(tree):
    with ReportTitle("kernel_list"):
        for kernel in tree.cuda_kernel_nodes:
            log(str(kernel))
