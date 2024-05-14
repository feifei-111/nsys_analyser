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

Build Logs Like:
<img width="915" alt="image" src="https://github.com/feifei-111/nsys_analyser/assets/79986504/057dbd4c-212e-4e12-993d-d4597460547f">

<img width="1060" alt="image" src="https://github.com/feifei-111/nsys_analyser/assets/79986504/9e4cbd89-a45c-4cd3-95a2-2990a053638a">
