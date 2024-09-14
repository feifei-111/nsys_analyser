# For analysing dumpped json from Nsys profiler

export nsys json file:
```bash
nsys export xxxxx.nsys-rep --type json --force-overwrite true -o xxxxx.json
```


```py
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

```

Build Logs Like:
```
threads: 2, main_thread_trees: 18

=============================================    kernel_ratio    =============================================
total_time: 6075.279597 ms, total_kernel_time: 0.017472  , kernel_ratio: 0.00
==============================================================================================================

===========================================    kernel_time_cost    ===========================================
total_kernel_time                        0.013966   ms
--------------------------------------------------------------------------------------------------------------
sync                                    :  kernel_cost = 0.008014   ms,  count = 1    , ratio = 57.38
memcpy                                  :  kernel_cost = 0.005952   ms,  count = 1    , ratio = 42.62
==============================================================================================================
```
