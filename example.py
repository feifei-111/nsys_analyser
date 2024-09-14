from nsys_analyser import NodeFilter, Report, ReportGuard, create_tree
from nsys_analyser.analyser import (
    analyse_kernel_ratio,
    analyse_kernel_time_cost,
)

json_path = "./test_memcpy.json"
tree = create_tree(json_path, NodeFilter.text_filter("step 0"))

report = Report("./log.log")

# just print
analyse_kernel_ratio(tree)

# dump to file
with ReportGuard(report):
    analyse_kernel_ratio(tree)
    analyse_kernel_time_cost(tree)
