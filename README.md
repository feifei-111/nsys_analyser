# For analysing dumpped json from Nsys profiler


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
    base_name = os.path.basename(log_path)

    under_line_count = 0
    for i in range(len(base_name)):
        idx = len(base_name) - i - 1
        if base_name[idx] == "_":
            under_line_count += 1
            if under_line_count == 2:
                print(base_name[0:idx])
                print(base_name[idx:])
                break

    analyse_interpreter_run(tree)

    print("\n")

    with line_printer(f"op status"):
        analyse_op_time_cost(tree)

    with line_printer(f"op's kernel status"):
        analyse_op_kernel_time_cost(tree)

    with line_printer(f"op list"):
        show_op_list(tree)

    with line_printer(f"op list"):
        show_kernel_list(tree)


sys.stdout = savedStdout
```
