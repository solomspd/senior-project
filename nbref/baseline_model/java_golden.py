import javalang
import pickle
final_tree = []


class Tree(object):
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.state = None
        self.idx = -1
        self.visited = False
        self.num_children = 0
        self.children = list()
        # self.childr = None

    def add_child(self, child):
        self.num_children += 1
        child.parent = self
        self.children.append(child)
        # if childl is not None:
    # def add_child_r(self, childr):
    # 	self.childr = childr
    # 	if childr is not None:
    # 		childr.parent = self
    # 		self.num_children += 1

    @staticmethod
    def get_root(node):
        if node.parent is None:
            return node
        else:
            return Tree.get_root(node.parent)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        # if getattr(self, '_depth'):
        #    return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        if(len(self.children) > 0):
            return self.children[0]


def create_tree(node):
    tree = Tree(node)
    if (node != None):
        for child in node.children:
            tree.add_child(node)


idx = 0
map2 = {'java.util.HashMap': 0, False: 1, 'java.util.Map': 2, 'Map': 3, 'Integer': 4, 'boolean': 5, 'int': 6, '=': 7, '>>': 8, '+': 9, '<': 10, 'length': 11, '==': 12, 'false': 13, '&&': 14, '&': 15, 'Math': 16, 'sqrt': 17, '<=': 18, '*': 19, '+=': 20, 'double': 21, 'System.out': 22, 'print': 23, '", "': 24, 'println': 25, '/': 26, 'printf': 27, 'String': 28, 'long': 29, 'null': 30, '-': 31, '!=': 32}

def test(node):
    count = 0
    global idx, map2

    tree = Tree(0)

    for i in range(0, len(node.children)):
        child = node.children[i]
        if type(child) == set:
            child = "set"
        elif type(child) == list and len(child) == 0:
            child = "list"
        # print(type(child))
        # newchild = Tree(0)
        if type(child) is None or (type(child) is not list and not ("javalang" in str(type(child)))):  # if not a list
            # tree.add_child( Tree(node.children[i]))
            # Tree("none" if type(child) is None else node.children[i]))
            tree.add_child(Tree(44 if child not in map2 else map2[child]))
            # tree.add_child(Tree(0 if type(child) is None else node.children[i]))

            # print(node.children[i],"case1")
            # print("case1")
        elif type(child) == list and len(child) == 0 and not ("javalang" in str(type(child))):  # handle empty list
            # tree.add_child(Tree(node.children[i]))
            tree.add_child(Tree(44 if child not in map2 else map2[child]))
            # print(node.children[i],"case2")

            # print(0)
            # tree.add_child(Tree(0))
        elif type(child) is set and not ("javalang" in str(type(child))):
            # tree.add_child(Tree(node.children[i]))
            # if child not in map:
            #     map[child] = idx
            #     idx += 1
            tree.add_child(Tree(44 if child not in map2 else map2[child]))
            # print(node.children[i],"case3")
            # tree.add_child(Tree(0))
            # print(node.children[i])
        else:  # >1 , repeat
            # print("PARENT", type(node.children[i]))
            if (type(child) is list):
                for j in range(0, len(child)):  # child is a list, create node for each
                    count += 1
                    if ("javalang" in str(type(child[j]))):
                        tree.add_child(test(child[j]))
                        # tree.add_child(Tree(child[j]))
            else:
                count += 1
                if ("javalang" in str(type(child))):
                    tree.add_child(test(child))
                    # tree.add_child(Tree(child))
    return tree


def get_golden(path):
    with open(path) as f:
        parser = javalang.parse.parse(f.read())
        # for path, node in parser:
        for i in range(0, 100):
            tree = test(parser)
            dict = {}
            dict['id'] = i
            dict['tree'] = tree
            dict['treelen'] = tree.num_children
            final_tree.append(dict)
        # with open("../data/re/tst_1/golden_c/samples.obj",'wb') as javafile:
        #    pickle.dump(final_tree,javafile)
    return final_tree


def create_tree_from_flat_list(node_list, index=1):
    if index >= len(node_list)-1 or node_list[index-1] is None:
        return None
    # pdb.set_trace()
    d = node_list[index-1]
    l = index * 2
    r = l + 1
    tree = Tree(d)
    left_child = create_tree_from_flat_list(node_list, l)
    right_child = create_tree_from_flat_list(node_list, r)

    if(left_child is not None):
        tree.add_child(left_child)
    if(right_child is not None):
        tree.add_child(right_child)
    return tree
