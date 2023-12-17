# For analysing dumpped json from Nsys profiler


'''py
import sys
from nsys_json.parse_json import create_nodes, create_tree
from nsys_json.analyser import analyse_llama


if __name__ == "__main__":

    if len(sys.argv) > 1:
        json_name = sys.argv[1]
    else:
        json_name = "sot"

    target_step = "15"

    nodes = create_nodes(json_name)
    root = create_tree(nodes, target_step)
    analyse_llama(root)
'''
