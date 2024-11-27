from nsys_analyser import NodeFilter, Report, ReportGuard, create_tree
from nsys_analyser.analyser import (
    analyse_kernel_coverage,
    analyse_kernel_status,
)

json_path = "./example.json"

# create call stack tree, with filters to local your target (use nvtx to capture)
# (NodeFilter.text_filter need the full name of a nvtx event)
tree = create_tree(json_path, NodeFilter.text_filter("step 0"))

# use analyse tools, they print the result
# you can add your analyser under dir `nsys_analyser/analyser`
analyse_kernel_coverage(tree)
analyse_kernel_status(tree, device=0)

# if you want dump to file, use `>` in bash
# or use ReportGuard if you want dump results to different files
report = Report("./log.log")
with ReportGuard(report):
    analyse_kernel_coverage(tree)
    analyse_kernel_status(tree, device=0)
