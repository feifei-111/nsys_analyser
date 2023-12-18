# For analysing dumpped json from Nsys profiler


```py
import sys
from nsys_analyser.parse_json import create_nodes, create_tree
from nsys_analyser.analyser import analyse_ops_with_multi_thread, show_op_list
from nsys_analyser.utils import line_printer


if __name__ == "__main__":
    target_step = "15"

    nodes = create_nodes("xxxxx.json")
    tree = create_tree(nodes, target_step, filter=lambda x: x.text == "llama")

    with line_printer("llama op status"):
        analyse_ops_with_multi_thread(tree)

    with line_printer("llama op list"):
        show_op_list(tree)
```
