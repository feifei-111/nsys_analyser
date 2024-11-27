from ..report import LINE_WIDTH, ReportTitle, log
from ..utils import DefaultDict as ddict
from ..utils import sort_on_values, target_events_checker


def analyse_kernel_coverage(tree):
    total_covered_time = sum(
        root.kernel_cover_time() for root in tree.main_roots
    )
    total_ranged_time = sum(
        root.kernel_ranged_time() for root in tree.main_roots
    )
    total_kernel_count = sum(len(root.kernels()) for root in tree.main_roots)

    with ReportTitle("KERNEL COVERAGE"):
        log(
            "ranged_time: {ranged_time:<10f} ms, kernel_covered: {kernel_covered:<10f}, kernel_count: {kernel_count:<4d}, ratio: {ratio:<.2f}".format(
                ranged_time=total_ranged_time / 1000000,
                kernel_covered=total_covered_time / 1000000,
                ratio=total_covered_time / total_ranged_time * 100,
                kernel_count=total_kernel_count,
            )
        )


def analyse_kernel_status(tree, device=0):
    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    total_ranged_time = sum(
        root.kernel_ranged_time() for root in tree.main_roots
    )
    for kernel in tree.cuda_kernel_nodes:
        if kernel.deviceID == device:
            kernel_time_cost[kernel.text] += kernel.time_cost
            kernel_count[kernel.text] += 1

    with ReportTitle(f"KERNEL STATUS (DEVICE{device})"):
        log(
            "total_ranged_time: {total_ranged_time:<10f} ms".format(
                total_ranged_time=total_ranged_time / 1000000
            )
        )
        log("-" * LINE_WIDTH)
        for k, v in sort_on_values(kernel_time_cost):
            log(
                "{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms,  count = {kernel_count:<5d}, ratio = {ratio:<.2f}".format(
                    k=k,
                    kernel_cost=v / 1000000,
                    kernel_count=kernel_count[k],
                    ratio=v / total_ranged_time * 100,
                )
            )


def analyse_kernel_status_under_target_events(
    tree, target_events=None, device=0
):
    checker = target_events_checker(target_events)
    total_time = sum(root.kernel_ranged_time() for root in tree.main_roots)

    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    kernel_time = 0

    for root in tree.main_roots:
        target_nodes = root.find_surface(checker)
        for node in target_nodes:
            for kernel in node.kernels():
                if kernel.deviceID == device:
                    kernel_time_cost[kernel.text] += kernel.time_cost
                    kernel_count[kernel.text] += 1
                    kernel_time += kernel.time_cost

    with ReportTitle(f"KERNEL STATUS UNDER {target_events} (DEVICE{device})"):
        log(
            "total_ranged_time: {total_ranged_time:<10f} ms, target kernel ratio: {ratio:<.2f}".format(
                total_ranged_time=total_time / 1000000,
                ratio=kernel_time / total_time * 100,
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
    with ReportTitle("KERNEL LIST"):
        for kernel in tree.cuda_kernel_nodes:
            log(str(kernel))
