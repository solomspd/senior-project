import inspect
import javalang
import sys
import javalangSelect
from copy import deepcopy
import networkx as nx
from pathlib import Path

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





		
	