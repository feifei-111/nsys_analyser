from ..report import LINE_WIDTH, ReportTitle, log
from ..utils import DefaultDict as ddict
from ..utils import sort_on_values, target_events_checker


_device_cache: dict[int, int] = {}


def detect_device(tree) -> int:
    tree_id = id(tree)
    if tree_id in _device_cache:
        return _device_cache[tree_id]
    devices = set(k.deviceID for k in tree.cuda_kernel_nodes)
    if len(devices) == 1:
        result = devices.pop()
        _device_cache[tree_id] = result
        return result
    raise ValueError(
        f"Multiple devices found: {devices}. Specify device explicitly."
    )


def analyse_kernel_coverage(tree, device=None):
    if device is None:
        device = detect_device(tree)
    total_covered_time = 0
    total_ranged_time = 0
    total_kernel_count = 0
    for root in tree.all_roots():
        cover = root.kernel_cover_time().get(device)
        if cover:
            total_covered_time += cover[0]
            total_ranged_time += cover[1]
        total_kernel_count += len(root.kernels().get(device, []))

    with ReportTitle(f"KERNEL COVERAGE (DEVICE{device})"):
        log(
            "ranged_time: {ranged_time:<10f} ms, kernel_covered: {kernel_covered:<10f}, kernel_count: {kernel_count:<4d}, ratio: {ratio:<.2f}".format(
                ranged_time=total_ranged_time / 1000000,
                kernel_covered=total_covered_time / 1000000,
                ratio=(
                    total_covered_time / total_ranged_time * 100
                    if total_ranged_time
                    else 0
                ),
                kernel_count=total_kernel_count,
            )
        )


def analyse_kernel_status(tree, device=None):
    if device is None:
        device = detect_device(tree)
    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    roots = list(tree.all_roots())
    total_ranged_time = 0
    for root in roots:
        cover = root.kernel_cover_time().get(device)
        if cover:
            total_ranged_time += cover[1]

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
                    ratio=(
                        v / total_ranged_time * 100 if total_ranged_time else 0
                    ),
                )
            )


def analyse_kernel_status_under_target_events(
    tree, target_events=None, device=None
):
    if device is None:
        device = detect_device(tree)
    checker = target_events_checker(target_events)
    roots = list(tree.all_roots())
    total_ranged_time = 0
    for root in roots:
        cover = root.kernel_cover_time().get(device)
        if cover:
            total_ranged_time += cover[1]

    kernel_time_cost = ddict(0)
    kernel_count = ddict(0)
    kernel_time = 0

    for root in roots:
        target_nodes = root.find_surface(checker)
        for node in target_nodes:
            for kernel in node.kernels().get(device, []):
                kernel_time_cost[kernel.text] += kernel.time_cost
                kernel_count[kernel.text] += 1
                kernel_time += kernel.time_cost

    with ReportTitle(f"KERNEL STATUS UNDER {target_events} (DEVICE{device})"):
        log(
            "total_ranged_time: {total_ranged_time:<10f} ms, target kernel ratio: {ratio:<.2f}".format(
                total_ranged_time=total_ranged_time / 1000000,
                ratio=(
                    kernel_time / total_ranged_time * 100
                    if total_ranged_time
                    else 0
                ),
            )
        )
        log("-" * LINE_WIDTH)
        for k, v in sort_on_values(kernel_time_cost):
            log(
                "{k:<40s}:  kernel_cost = {kernel_cost:<10f} ms,  count = {kernel_count:<5d}, ratio = {ratio:<.2f}".format(
                    k=k,
                    kernel_cost=v / 1000000,
                    kernel_count=kernel_count[k],
                    ratio=(
                        v / total_ranged_time * 100 if total_ranged_time else 0
                    ),
                )
            )


def analyse_kernel_list(tree):
    with ReportTitle("KERNEL LIST"):
        for kernel in tree.cuda_kernel_nodes:
            log(str(kernel))
