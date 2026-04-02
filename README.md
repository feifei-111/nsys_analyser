# For analysing dumpped json from Nsys profiler

export nsys json file:
```bash
nsys export xxxxx.nsys-rep --type json --force-overwrite true -o xxxxx.json

# for new version
nsys export xxxx.nsys-rep --type=jsonlines --force-overwrite true -o xxxx.jsonl
```

use this tool for analysing:
see [example.py](example.py)

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
