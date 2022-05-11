import argparse
import torch
from pathlib import Path

from model.model_top import cavaj
from data_proc import __load_bytecode

import inspect
import javalang
import sys
import javalangSelect
from copy import deepcopy
import networkx as nx

type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] 
# print(type_map)

def generateCode(ast, file):
    type_map_code = []
    codeConv = []
    code = []
    type_map_genCode = list(nx.dfs_preorder_nodes(ast, source=0))
    for item in type_map_genCode:
        codeline = code.append(type_map[item])
        code.append(javalangSelect.ret_string(codeline))
    return code 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('--model', type=str, metavar='checkpoint', nargs=1)
    parser.add_argument('--llc', type=str, metavar='input file', nargs='+')
    arg = parser.parse_args()

    checkpoint = torch.load(arg.model, map_location='cpu')
    model = cavaj()
    model.load_state_dict(checkpoint['model_state_dict'])

    for i in arg.llc:
        llc_path = Path(i)
        with open (llc_path) as file:
            try:
                llc = __load_bytecode(file)
            except Exception as e:
                print(f'Failed to load Low Level Code file due to {e}')
                continue
        pred = model(llc)
        