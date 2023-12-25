# For analysing dumpped json from Nsys profiler

export nsys json file:
```bash
nsys export xxxxx.nsys-rep --type json --force-overwrite true -o xxxxx.json
```


```py
import sys, os
from nsys_analyser.parse_json import create_tree
from nsys_analyser.analyser import *
from nsys_analyser.utils import line_printer


target_step = "55"

assert len(sys.argv) > 2

json_path = sys.argv[1]
log_path = sys.argv[2]

savedStdout = sys.stdout
tree = create_tree(json_path, target_step, "forward")

with open(log_path, "w") as file:
    sys.stdout = file

    analyse_interpreter_run(tree)
    print("\n")

    with line_printer(f"op status"):
        analyse_op_time_cost(tree)

    with line_printer(f"kernel status"):
        analyse_kernel_time_cost(tree)

    with line_printer(f"kernel list"):
        show_kernel_list(tree)

    with line_printer(f"op list"):
        show_op_list(tree)

sys.stdout = savedStdout
```