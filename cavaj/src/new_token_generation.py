import inspect
from copy import deepcopy
import javalang
from pathlib import Path


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
                    # print(inv)
                    if check_if_jl_class(type(inv)):
                        to_invoke.append(inv)
                    elif type(inv) == list and len(inv) != 0 and check_if_jl_class(type(inv[0])):
                        for i in inv:
                            to_invoke.append(i)
                    else:
                        if inv is not None and inv:
                            all_node_info.append(str(inv))
    return all_node_info, to_invoke
               
class node:
    def __init__(self):
        self.atomic_vals = [] #filter list
        self.to_invoke = []
        self.children =[]
        self.parent =[]
        self.visited = False
    #a way to link
    
def traverse_tree(instance, flag):
    global ast_code
    global ff
    if instance.visited or (len(instance.to_invoke) == 0 and type(instance) != None):  # Base case
        return
    else:
        # first_child = node()
        instance.visited = True
        for x in instance.to_invoke: # one javalang class
            child = node()
            if check_if_jl_class(type(x)):
                child.atomic_vals, child.to_invoke = extract_node_info(x)
                for i in child.atomic_vals:
                    if(i):
                        ast_code.append(i)                        
                        # ast_code.append(' ')
                ff.write("%s\n" % str([type(x),child.atomic_vals]))
                             
                       
                tokens_file.write('\n') 
                instance.children.append(child)
                traverse_tree(child, True)


if __name__ == "__main__":
    ast_code = []
    src_path = '/home/ramy/andrew/senior-project/cavaj/data/50k/java_src'
    files = [str(item) for item in Path(src_path).iterdir() if item.is_file()]
    
    for file in files:
        with open(file, 'r') as fileNew:
            try:
                print(file)
                data = fileNew.read()
                parser = javalang.parse.parse(data) 
                # print(parser)      
                x, y = extract_node_info(parser.children[2][0])
                root = node()
                root.atomic_vals = x
                root.to_invoke = y
                ast_code.append(root.atomic_vals)
                tokens_file = open("new_tokens.txt",'a+')
                # tokens_file_write = open("test.txt",'w')
                for value in root.atomic_vals:               
                    # if value not in tokens_file.read():
                    tokens_file.write(value)                   
                    # tokens_file.write(' ') 
                tokens_file.write('\n')               
                mytree = root  
                traverse_tree(root, False)
            except Exception as e:
                print(e)
    

# for i in ast_code:
#     print(i)