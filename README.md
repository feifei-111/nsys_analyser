# For analysing dumpped json from Nsys profiler


```py
import sys
from nsys_analyser.parse_json import create_nodes, create_tree
from nsys_analyser.analyser import analyse_llama


if __name__ == "__main__":
    target_step = "15"
    nodes = create_nodes("xxxxx.json")
    root = create_tree(nodes, target_step)
    with line_printer("llama op status"):
        analyse_ops_under_named_event(root, "llama")

    with line_printer("llama op list"):
        show_op_list(root, "llama")
```
