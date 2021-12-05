import javalang 

class Tree(object):
	def __init__(self,value):
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
	#if getattr(self, '_depth'):
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
		if(len(self.children)>0):
			return self.children[0]

def create_tree(node):  
    tree = Tree(node)
    if (node!=None):
        for child in node.children:
            tree.add_child(node)
                

count = 0
def test(node):
    global count 
   
    tree = Tree(0)

    for i in range(0, len(node.children)):
        child = node.children[i]
        #print(type(child))
        # newchild = Tree(0)
        if type(child) is None or (type(child) is not list  and not ("javalang" in str(type(child)))): #if not a list
                tree.add_child( Tree(node.children[i]))
                # print("case1")
                #print(node.children[i])
        elif type(child) == list and len(child)== 0 and not ("javalang" in str(type(child))): #handle empty list
                tree.add_child( Tree(node.children[i]))
                #print(node.children[i])
        elif type(child) is set and not ("javalang" in str(type(child))):
                tree.add_child( Tree(node.children[i]))
                #print(node.children[i])
        else: #>1 , repeat
            #print("PARENT", type(node.children[i])) 
            if (type(child) is list):
                for j in range(0,len(child)): #child is a list, create node for each
                    count+=1
                    if ("javalang" in str(type(child[j]))):
                        tree.add_child(test(child[j]))
                        #tree.add_child(Tree(child[j]))
            else:
                count+=1
                if ("javalang" in str(type(child))):
                    tree.add_child(test(child))
                    #tree.add_child(Tree(child))
    return tree

with open("prime_source.txt") as f:
    parser = javalang.parse.parse(f.read())
    # for path, node in parser:
    tree = test(parser)
        
        
        
            


            
def create_tree_from_flat_list(node_list, index=1):
	if index >= len(node_list)-1 or node_list[index-1] is None:
		return None
	# pdb.set_trace()
	d = node_list[index-1]
	l = index * 2
	r = l + 1
	tree = Tree(d)
	left_child  = create_tree_from_flat_list(node_list, l)
	right_child = create_tree_from_flat_list(node_list, r)

	if(left_child is not None):
		tree.add_child(left_child)
	if(right_child is not None):
		tree.add_child(right_child)
	return tree






