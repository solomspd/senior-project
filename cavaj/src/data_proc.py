from tqdm import tqdm
from pathlib import Path
import networkx as nx
import dgl
import javalang
import sys,inspect
import regex as re

class data_proc:

	def __init__(self):
		self.exclude = set()
		self.trg = []
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
		self.ast = nx.graph()
		self.src_f,self.src_g = [],[]

	def load_src(self, path):
		for i in tqdm(sorted(path.iterdir())):
			with open(i) as file:
				try:
					self.trg.append(self.proc_ast(file))
				except:
					self.exclude.add(i)
    				
	def load_llc(self, path):
		ii = 0
		for i in tqdm(sorted(path.iterdir())):
			with open(i) as llc_file:
				if i.stem in self.exclude: continue
				try:
					ga,nf,ng = self.load_bytecode([line.strip() for line in llc_file.readlines()])
				except:
					del self.trg[ii]
					continue
				ii += 1
				self.src_f.append(nf)
				self.src_g.append(ng)
		pass

	def __proc_ast(self, in_file):
		parsed_src = javalang.parse.parse(in_file.read())
		return self.__propagate_ast(parsed_src.types[0])

	def __propagate_ast(self, node):
		if type(node) is list and len(node) == 1:
			node = node[0]
		if 'body' in node.attrs and node.body is not None:
			if type(node.body) is list:
				for i in node.body:
					self.ast.add_child(self.__propagate_ast(i))
			else:
				self.tree.add_child(self.__propagate_ast(node.body))
		else:
			if 'expression' in node.attrs and node.expression is not None:
				self.tree.add_child(self.__propagate_ast(node.expression))

	def load_bytecode(lines):
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

