import inspect
from copy import deepcopy
import javalang
from pathlib import Path
import networkx as nx
import sys
# import javalangSelect.py

ff = open('dummy.txt', 'w')
def check_if_jl_class(check_this):
    for name, obj in inspect.getmembers(javalang.tree):
        if check_this == obj:
            return True
    return False


def extract_node_info(instance):
    all_node_info = ""
    # print(instance)
    # print(type(instance))
    all_node_info = []
    to_invoke = []
    if check_if_jl_class(type(instance)):
        for i in instance.attrs:
            if i != 'documentation':
                if i[0] != '_':  # Not builtin attr (--> get only javalang specific)
                    # print(i, ":::")
                    inv = getattr(instance, i)
                    
                    if check_if_jl_class(type(inv)):
                        to_invoke.append(inv)
                    elif type(inv) == list and len(inv) != 0 and check_if_jl_class(type(inv[0])):
                        for i in inv:
                            to_invoke.append(i)
                    else:
                        if inv is not None and inv:
                            all_node_info.append(type(instance))
                            # ff.write("%s\n" % type(instance))
    return all_node_info, to_invoke
               
class node:
    def __init__(self):
        self.atomic_vals = [] #filter list
        self.to_invoke = []
        self.children =[]
        self.parent =[]
        self.visited = False
    #a way to link
cur_idx=0
type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] 
parent=0
# child_idx=0

def traverse_tree(instance, flag,parent):
    global ast_code
    global ff
    global cur_idx
    
    # global parent
    global type_map
    # global child_idx
    # cur_idx+=1
    #type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
    if instance.visited or (len(instance.to_invoke) == 0 and type(instance) != None):  # Base case
        # parent+=1
        return
    else:
        # first_child = node()
        # child_idx+=1
        
        instance.visited = True
        for x in instance.to_invoke: # one javalang class
            child = node()
            if (type(x)):
                child.atomic_vals, child.to_invoke = extract_node_info(x)
                for i in child.atomic_vals:
                    if(i):
                        cur_idx+=1
                        ast.add_node(cur_idx, type=type_map.index((type(x))))
                        ast.add_edge(parent, cur_idx)

                if len(child.atomic_vals[1:]) != 0:
                    temp_list = [type(x), child.atomic_vals[1:]]
                else:
                    temp_list = [type(x)]
                ff.write("%s\n" % str(temp_list))     
                #tokens_file.write('\n') 
                # instance.children.append(child)
                traverse_tree(child, True, cur_idx)
                # parent+=1
# ast = nx.Graph()

def get_root(fileNew):
# if __name__ == "__main__":
    ast_code = []
    global ast 
    ast = nx.Graph()
    global type_map
    # src_path = '/home/ramy/andrew/senior-project/cavaj/data/50k/java_src'
    # files = [str(item) for item in Path(src_path).iterdir() if item.is_file()]
    # file = src_path
    # for file in files:
    # with open(file, 'r') as fileNew:

    # print(file)
    data = fileNew.read()
    parser = javalang.parse.parse(data) 
    x, y = extract_node_info(parser.children[2][0]) #change
    root = node()
    root.atomic_vals = x
    root.to_invoke = y
    ast_code.append(root.atomic_vals)
    # tokens_file = open("new_tokens.txt",'a+')
    # tokens_file_write = open("test.txt",'w')
    # for value in root.atomic_vals:               
    #     # if value not in tokens_file.read():
    #     tokens_file.write(str(value))   
    #     #print(value)            
    #     # tokens_file.write(' ') 
    # tokens_file.write('\n')               
    mytree = root 
    cur_idx = 0
    ast.add_node(cur_idx, type=type_map.index(root.atomic_vals[0]))
    traverse_tree(root, False,0)
    return ast


# get_root('/home/g07/senior-project/cavaj/data/50k/java_src/AABBPool.java')
# print('done')