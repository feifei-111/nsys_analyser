from nsys_analyser import NodeFilter, Report, ReportGuard, create_tree
from nsys_analyser.analyser import (
    analyse_kernel_ratio,
    analyse_kernel_time_cost,
)

# create call stack tree, with filters to local your target (use nvtx to capture)
json_path = "./example.json"
tree = create_tree(json_path, NodeFilter.text_filter("step 0"))

# just print
analyse_kernel_ratio(tree)
analyse_kernel_time_cost(tree)

# dump to file
report = Report("./log.log")
with ReportGuard(report):
    analyse_kernel_ratio(tree)
    analyse_kernel_time_cost(tree)
