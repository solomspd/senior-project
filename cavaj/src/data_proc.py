from tqdm import tqdm
from pathlib import Path
import networkx as nx
import dgl
import javalang
import sys,inspect
import re

class data_proc:

	def __init__(self, arg):
		self.exclude = set()
		self.ground_truth = []
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
		self.src_f,self.src_g,self.src_l = [],[],[]
		self.ast_idx = 0
		self.arg = arg
	
	def load_data(self, src_path, llc_path):
		self.__load_src(src_path)
		self.__load_llc(llc_path)
		print("Dataset size:", len(self.ground_truth), "\nRejected files:", self.arg.data_point_num - len(self.ground_truth))
		return self.ground_truth, self.src_l, self.src_g, self.src_f # TODO more descriptive name

	def __load_src(self, path):
		for i in tqdm(sorted(path.iterdir())[:self.arg.data_point_num], desc="Loading java source"):
			with open(i) as file:
				try:
					self.ground_truth.append(self.__proc_ast(file))
				except:
					self.exclude.add(i)
    				
	def __load_llc(self, path):
		ii = 0
		for i in tqdm(sorted(path.iterdir())[:self.arg.data_point_num], desc="Loading bytecode"):
			with open(i) as llc_file:
				if i.stem in self.exclude: continue
				try:
					ga,nf,ng = self.__load_bytecode(llc_file)
				except:
					del self.ground_truth[ii]
					continue
				ii += 1
				self.src_f.append(nf)
				self.src_g.append(ng)
				self.src_l.append(ga)
		pass

	def __proc_ast(self, in_file):
		parsed_src = javalang.parse.parse(in_file.read())
		self.ast = nx.Graph()
		self.ast_idx = 0
		self.__propagate_ast(None, parsed_src.types[0])
		return dgl.from_networkx(self.ast)

	def __propagate_ast(self, parent, node):
		if type(node) is list and len(node) == 1:
			node = node[0]
		cur_idx = self.ast_idx
		if parent is not None:
			self.ast.add_node(cur_idx, type=self.type_map.index(type(node)))
			self.ast.add_edge(parent, cur_idx)
		self.ast_idx += 1
		if 'body' in node.attrs and node.body is not None:
			if type(node.body) is list:
				for i in node.body:
					self.__propagate_ast(cur_idx, i)
			else:
				self.__propagate_ast(cur_idx, node.body)
		else:
			if 'expression' in node.attrs and node.expression is not None: #TODO exhaustively cover all possible nodes
				self.__propagate_ast(cur_idx, node.expression)

	def __load_bytecode(self, llc_file):
		lines = [line.strip() for line in llc_file.readlines()]
		address_mapper = {}
		sameloc_mapper = {}
		instruction_identifier = {}
		identifier = 0
		data_edge_type = 0
		control_edge_type = 1
		instruction_edge_type = 2
		ft_fn_type = -1
		ft_cls_type = -2
		ft_ar_type = -3
		classes = []
		instructions = []
		functions = []
		graph = nx.DiGraph()
		idx = 0
		identifier = 0    
		feat = []
		edges = [] # format: (source, edge type, target)
		for line in lines:
			if line:
				
				#class
				if re.findall("class [a-zA-Z0-9_&$]*", line):
					className = re.findall("\b(?:class )\b|(\w+)", line)[1]
					
					graph.add_node(idx)
					classes.append(idx)
					feat.append(ft_cls_type)
					idx += 1

				#Functions
				if re.findall(".+({.*}|\(.*\));",line):
					functionName = re.findall("([a-zA-Z0-9_&$]*?)\(",line)
					args = []
					instructions.clear()
					if('(' in line):
						args = line[line.find('(')+1: line.find(')')].split(',')
					if(functionName):
						graph.add_edge(classes[-1], idx, e_type=control_edge_type, n_type = -2)
						edges.append((classes[-1], control_edge_type, idx))
						feat.append(ft_fn_type)
						f_idx = idx
						idx += 1

					type = re.findall("(public|private|protected)",line)
					if type:
						graph.add_edge(f_idx, idx, e_type=instruction_edge_type, n_type = -2)
						edges.append((f_idx, instruction_edge_type, idx))
						feat.append(ft_fn_type)
						idx += 1

					if (functionName): 
						returnType = re.findall("(?<=\s)(.*?)(?=\s{1,}%s)"%functionName,line)
						if returnType:
							graph.add_edge(f_idx, idx, e_type=instruction_edge_type, n_type = -2)
							edges.append((f_idx, instruction_edge_type, idx))
							feat.append(ft_fn_type)
							idx += 1
					
					for arg in args:
						if arg:
							graph.add_edge(f_idx, idx, e_type=instruction_edge_type, n_type = -2)
							edges.append((f_idx, instruction_edge_type, idx))
							feat.append(ft_fn_type)
							idx += 1
					functions.append(f_idx)
						
		#instructions
				elif re.findall("\d+[:]\s\w+",line):
					instructionInfo = re.findall("[^\s\\:\\\\\/\/<>.\'\"(),;]\w{0,}",line)
					instructionName = instructionInfo[1]
					if(instructionName not in instruction_identifier):
						identifier = identifier + 1
						instruction_identifier[instructionName] = identifier

					k = instruction_identifier[instructionName]
					I_idx = idx
					if(instructions):
						graph.add_edge(instructions[-1], idx, e_type=control_edge_type, n_type = k)
						edges.append((instructions[-1], control_edge_type, idx))
						idx += 1
					else:
						graph.add_edge(functions[-1], idx, e_type=control_edge_type, n_type = k)
						edges.append((functions[-1], control_edge_type, idx))
						idx += 1
					feat.append(k)
					if(len(instructionInfo) > 2):
						if(re.findall("#{0,1}\d+", instructionInfo[2])):
							graph.add_edge(idx, I_idx, e_type=instruction_edge_type, n_type = 0)
							edges.append((idx, instruction_edge_type,I_idx))
							feat.append(ft_ar_type)
							idx += 1
							if(instructionInfo[2][0] == '#'):
								if instructionInfo[2] in sameloc_mapper:
									graph.add_edge(sameloc_mapper[instructionInfo[2]], idx, e_type = control_edge_type, n_type = -1)
								
								sameloc_mapper[instructionInfo[2]] = idx
								comment = re.findall("(?<=\/\/ ).*[^;]", line)[0]
								address_mapper[instructionInfo[2]] = comment
					if(len(instructionInfo) > 3):
						if(re.findall("#{0,1}\d+", instructionInfo[3])):
							graph.add_edge(idx, I_idx, e_type=instruction_edge_type, n_type = -1)
							edges.append((idx, instruction_edge_type, I_idx))
							feat.append(ft_ar_type)
							idx += 1
					instructions.append(I_idx)
		return dgl.from_networkx(graph, edge_attrs = ['e_type','n_type']), feat, edges