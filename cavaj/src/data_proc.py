import inspect
import re
import sys
import logging
from pathlib import Path
import matplotlib
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import networkx as nx 
import javalang
import random
from tqdm import tqdm


class data_proc(Dataset):

	def __init__(self, arg, src_path, llc_path, cache_path=None):
		self.llc_path = llc_path
		self.src_path = src_path
		self.arg = arg
		self.exclude = set()
		self.instruction_identifier = ['aaload', 'aastore', 'aconst_null', 'aload', 'aload_0', 'aload_1', 'aload_2', 'aload_3', 'anewarray', 'areturn', 'arraylength', 'astore', 'astore_0', 'astore_1', 'astore_2', 'astore_3', 'athrow', 'baload', 'bastore', 'bipush', 'breakpoint', 'caload', 'castore', 'checkcast', 'd2f', 'd2i', 'd2l', 'dadd', 'daload', 'dastore', 'dcmpg', 'dcmpl', 'dconst_0', 'dconst_1', 'ddiv', 'dload', 'dload_0', 'dload_1', 'dload_2', 'dload_3', 'dmul', 'dneg', 'drem', 'dreturn', 'dstore', 'dstore_0', 'dstore_1', 'dstore_2', 'dstore_3', 'dsub', 'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2', 'f2d', 'f2i', 'f2l', 'fadd', 'faload', 'fastore', 'fcmpg', 'fcmpl', 'fconst_0', 'fconst_1', 'fconst_2', 'fdiv', 'fload', 'fload_0', 'fload_1', 'fload_2', 'fload_3', 'fmul', 'fneg', 'frem', 'freturn', 'fstore', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3', 'fsub', 'getfield', 'getstatic', 'goto', 'goto_w', 'i2b', 'i2c', 'i2d', 'i2f', 'i2l', 'i2s', 'iadd', 'iaload', 'iand', 'iastore', 'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4', 'iconst_5', 'idiv', 'if_acmpeq', 'if_acmpne', 'if_icmpeq', 'if_icmpge', 'if_icmpgt', 'if_icmple', 'if_icmplt', 'if_icmpne', 'ifeq', 'ifge', 'ifgt', 'ifle', 'iflt', 'ifne', 'ifnonnull', 'ifnull', 'iinc', 'iload', 'iload_0', 'iload_1', 'iload_2', 'iload_3', 'impdep1', 'impdep2', 'imul', 'ineg', 'instanceof', 'invokedynamic', 'invokeinterface', 'invokespecial', 'invokestatic', 'invokevirtual', 'ior', 'irem', 'ireturn', 'ishl', 'ishr', 'istore', 'istore_0', 'istore_1', 'istore_2', 'istore_3', 'isub', 'iushr', 'ixor', 'jsr†', 'jsr_w†', 'l2d', 'l2f', 'l2i', 'ladd', 'laload', 'land', 'lastore', 'lcmp', 'lconst_0', 'lconst_1', 'ldc', 'ldc_w', 'ldc2_w', 'ldiv', 'lload', 'lload_0', 'lload_1', 'lload_2', 'lload_3', 'lmul', 'lneg', 'lookupswitch', 'lor', 'lrem', 'lreturn', 'lshl', 'lshr', 'lstore', 'lstore_0', 'lstore_1', 'lstore_2', 'lstore_3', 'lsub', 'lushr', 'lxor', 'monitorenter', 'monitorexit', 'multianewarray', 'new', 'newarray', 'nop', 'pop', 'pop2', 'putfield', 'putstatic', 'ret†', 'return', 'saload', 'sastore', 'sipush', 'swap', 'tableswitch', 'wide', '(no name)']
		files = [str(item) for item in Path(self.src_path).iterdir() if item.is_file()]
		self.tokens = []
		self.counter = 1
		self.type_map =[]
		try:
			for file in files:
				with open(file, 'r') as fileNew:
					self.data = fileNew.read()
					self.tokens += list(javalang.tokenizer.tokenize(self.data))
				for token in self.tokens:
					self.type_map.append(token.value)
		except:
			self.counter+=1
			print(self.counter)
		self.type_map = list(dict.fromkeys(self.type_map))
		self.type_map = [x for x in self.type_map if (x.find('"') and not x[0].isdigit())]
		with open('tokens.txt', 'w') as testFile:
			for item in self.type_map:
				testFile.write("%s\n" % item)
		#self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
		self.trg_ast,self.trg_llc = [],[]
		self.ast_idx = 0
		self.cache_path = Path('tmp/') if cache_path is None else cache_path
		self.num_data_points = 0
		self.cache_path = cache_path / 'processed'
		super().__init__(cache_path)
	
	def get(self, idx):
		ast_load = torch.load(self.cache_path / f"ast_cache_{idx}.pt")
		llc_load = torch.load(self.cache_path / f"llc_cache_{idx}.pt")
		return ast_load,llc_load

	def hierarchy_pos(self,G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
		if not nx.is_tree(G):
			raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

		if root is None:
			if isinstance(G, nx.DiGraph):
				root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
			else:
				root = random.choice(list(G.nodes))

	def _hierarchy_pos(self,G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
		if pos is None:
			pos = {root:(xcenter,vert_loc)}
		else:
			pos[root] = (xcenter, vert_loc)
		children = list(G.neighbors(root))
		if not isinstance(G, nx.DiGraph) and parent is not None:
			children.remove(parent)  
		if len(children)!=0:
			dx = width/len(children) 
			nextx = xcenter - width/2 - dx/2
			for child in children:
				nextx += dx
				pos = self._hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
									vert_loc = vert_loc-vert_gap, xcenter=nextx,
									pos=pos, parent = root)
			return pos       
		return self._hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)	
	
	def process(self):
		ohd_llc = T.OneHotDegree(len(self.instruction_identifier), cat=False)

		for llc,ast in tqdm(self.raw_paths, desc="Loading dataset", total=self.arg.data_point_num, disable=self.arg.no_prog):
			with open(ast) as file:
				try:
					trg_ast = self.__proc_ast(file)
				except Exception as e:
					logging.warning(f"{ast} failed to import AST due to {e}")
					continue
			with open(llc) as file:
				try:
					trg_llc = self.__load_bytecode(file)
				except Exception as e:
					logging.warning(f"{llc} failed to import LLC due to {e}")
					continue
			
			ohd_llc(trg_llc)
			torch.save(trg_ast, self.cache_path / f"ast_cache_{self.num_data_points}.pt")
			torch.save(trg_llc, self.cache_path / f"llc_cache_{self.num_data_points}.pt")
			self.num_data_points += 1

		n_rejected = self.arg.data_point_num - self.num_data_points

		if n_rejected > 0:
			logging.warning(f"{n_rejected} files rejected")
		if self.num_data_points == 0:
			logging.error("All files rejected")
			raise Exception("All files rejected")
	
	def __reduce_to_actions(self, ast):
		# Returns an array of pairs of (node type, node parent, node index). in other words, the sequence of actions required to construct the graph
		edge_alias = {0:0}
		edge_index = 1
		ret_que = [[ast.nodes[0]["type"], 0, 0]] # len(token array) = SOS. parent = 0 because it does not have a parent and this 0 should never be used
		bfs = nx.bfs_predecessors(ast, 0)
		for i in bfs:
			if i[0] not in edge_alias:
				edge_alias[i[0]] = edge_index
				edge_index += 1
			ret_que.append([ast.nodes[i[0]]["type"], edge_alias[i[1]], edge_alias[i[0]]])
		ret_que.append([len(self.type_map) + 1, 0, 0]) # len(token array) + 1 = EOS. parent = 0 because it does not have a parent and this 0 should never be used
		ret_que = torch.Tensor(ret_que).long()
		return ret_que
	
	def len(self):
		return self.num_data_points

	@property
	def raw_paths(self):
		llc_iter = sorted(self.llc_path.iterdir())[:self.arg.data_point_num]
		src_iter = sorted(self.src_path.iterdir())[:self.arg.data_point_num]
		self.arg.data_point_num = min(len(llc_iter), len(src_iter))
		return zip(llc_iter, src_iter)
	
	@property
	def processed_file_names(self):
		return [] # TODO: add cache list here when 100% sure we're done with data handing. leaving this empty so we are forced to redo the entire cache every time we run the model
	
	def __proc_ast(self, in_file):
		parsed_src = javalang.parse.parse(in_file.read())
		self.ast = nx.Graph()
		self.ast_idx = 0
		self.ast.add_node(self.ast_idx, type=self.type_map.index("class"))
		self.__propagate_ast(None, parsed_src.types[0])
		pos = self.hierarchy_pos(self.ast,1)
		nx.draw(self.ast, pos=pos, with_labels=True, node_size=5,font_size=0.01)
		plt.savefig('hierarchy.png', dpi=1000)
		ast_ret = from_networkx(self.ast, group_node_attrs=['type'])
		ast_ret.x = torch.cat((ast_ret.x, torch.tensor([[len(self.type_map) + 1]]))) # adding EOS (len(token array) + 1) token to ground truth
		return self.__reduce_to_actions(self.ast), ast_ret

	def addToTree(self, parent, cur_idx, str1):
		if type(str1) != javalang.tree.VariableDeclarator and str1 is not None and str1:
			self.ast.add_node(cur_idx, type=self.type_map.index(str1))
			self.ast.add_edge(parent, cur_idx)

	def __propagate_ast(self, parent, node):
		if type(node) is list and len(node) == 1:
			node = node[0]
		cur_idx = self.ast_idx
		listAttr = node.attrs
		self.str1 = ""
		if parent is not None:
			if type(node.children) is list and len(node.children) > 0: 
				for newChild in node.children:
					if newChild is not None and newChild:
						self.strType = str(type(newChild))
						if self.strType == "javalang.tree.Class":  # For Classes
							for InNewChild in newChild.children:
								for allInNewChild in InNewChild.children:
									str1 = allInNewChild
									self.addToTree(parent, cur_idx, str1)
						elif "javalang.tree" in self.strType: # For any type of Javalang
							for allInNewChild in newChild.children:
								str1 = allInNewChild
								self.addToTree(parent, cur_idx, str1)
						else: # others
							if type(newChild) == str:
								str1 = newChild
								self.addToTree(parent, cur_idx, str1)
							else:
								for allInNewChild in newChild:
									str1 = allInNewChild
									self.addToTree(parent, cur_idx, str1)
		self.ast_idx += 1
		for attr in node.attrs:
			#print(vars(node))
			#print(vars(node)[attr])
			if vars(node)[attr] is not None:
				if type(vars(node)[attr]) is list:
					for i in vars(node)[attr]:
						self.__propagate_ast(cur_idx, i)			

	def __load_bytecode(self, llc_file):
		lines = [line.strip() for line in llc_file.readlines()]
		address_mapper = {}
		sameloc_mapper = {}
		control_edge_type = 1
		instruction_edge_type = 2
		fn_placeholder = 2
		inst_placeholder = 3
		classes = []
		instructions = []
		functions = []
		graph = nx.DiGraph()
		idx = 0
		for line in lines:
			if line:
				
				#class
				if re.findall("class [a-zA-Z0-9_&$]*", line):
					className = re.findall("\b(?:class )\b|(\w+)", line)[1]
					graph.add_node(idx, n_type=1)
					classes.append(idx)
					idx += 1

				#Functions
				if re.findall(".+({.*}|\(.*\));",line):
					functionName = re.findall("([a-zA-Z0-9_&$]*?)\(",line)
					args = []
					instructions.clear()
					if('(' in line):
						args = line[line.find('(')+1: line.find(')')].split(',')
					if(functionName):
						graph.add_node(idx, n_type = fn_placeholder)
						graph.add_edge(classes[-1], idx, e_type=control_edge_type)
						f_idx = idx
						idx += 1

					type = re.findall("(public|private|protected)",line)
					if type:
						graph.add_node(idx, n_type = fn_placeholder)
						graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
						idx += 1

					if (functionName): 
						returnType = re.findall("(?<=\s)(.*?)(?=\s{1,}%s)"%functionName,line)
						if returnType:
							graph.add_node(idx, n_type = fn_placeholder)
							graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
							idx += 1
					
					for arg in args:
						if arg:
							graph.add_node(idx, n_type = fn_placeholder)
							graph.add_edge(f_idx, idx, e_type=instruction_edge_type)
							idx += 1
					functions.append(f_idx)
						
		#instructions
				elif re.findall("\d+[:]\s\w+",line):
					instructionInfo = re.findall("[^\s\\:\\\\\/\/<>.\'\"(),;]\w{0,}",line)
					instructionName = instructionInfo[1]

					k = self.instruction_identifier.index(instructionName) + 4
					I_idx = idx
					if(instructions):
						graph.add_node(idx, n_type = k)
						graph.add_edge(instructions[-1], idx, e_type=control_edge_type)
						idx += 1
					else:
						graph.add_node(idx, n_type = k)
						graph.add_edge(functions[-1], idx, e_type=control_edge_type)
						idx += 1
					if(len(instructionInfo) > 2):
						if(re.findall("#{0,1}\d+", instructionInfo[2])):
							graph.add_node(idx, n_type = 0)
							graph.add_edge(idx, I_idx, e_type=instruction_edge_type)
							idx += 1
							if(instructionInfo[2][0] == '#'):
								if instructionInfo[2] in sameloc_mapper:
									graph.add_node(sameloc_mapper[instructionInfo[2]], n_type = instruction_edge_type)
									graph.add_edge(sameloc_mapper[instructionInfo[2]], idx, e_type = control_edge_type)
								
								sameloc_mapper[instructionInfo[2]] = idx
								comment = re.findall("(?<=\/\/ ).*[^;]", line)[0]
								address_mapper[instructionInfo[2]] = comment
					if(len(instructionInfo) > 3):
						if(re.findall("#{0,1}\d+", instructionInfo[3])):
							graph.add_node(idx, n_type = instruction_edge_type)
							graph.add_edge(idx, I_idx, e_type=instruction_edge_type)
							idx += 1
					instructions.append(I_idx)
		return from_networkx(graph, group_node_attrs=["n_type"], group_edge_attrs=["e_type"])