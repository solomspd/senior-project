import inspect
import re
import sys
from pathlib import Path

from torch_geometric.utils.convert import from_networkx
import javalang
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import Dataset
import pickle


class data_proc(Dataset):

	def __init__(self, arg, src_path, llc_path, cache_path=None):
		super.__init__()
		self.exclude = set()
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
		self.trg_ast,self.trg_llc = [],[]
		self.ast_idx = 0
		self.llc_path = llc_path
		self.src_path = src_path
		self.cache_path = Path('tmp/') if cache_path is None else cache_path
		self.arg = arg
		self.num_data_points = 0
	
	def get(self, idx):
		ast_load = pickle.load(self.cache_path / f"ast_cache_{idx}.pkl")
		llc_load = pickle.load(self.cache_path / f"llc_cache_{idx}.pkl")
		return ast_load,llc_load
	
	def process(self):
		for i in tqdm(self.raw_paths, desc="Loading dataset"):
			with open(i[0]) as file:
				try:
					trg_ast = self.__proc_ast(file)
				except:
					continue
			with open(i[1]) as file:
				try:
					trg_llc = self.__load_bytecode(file)
				except:
					continue
			with open(self.cache_path / f"ast_cache_{self.num_data_points}.pkl", 'wb') as file:
				pickle.dump(trg_ast, file)
			with open(self.cache_path / f"llc_cache_{self.num_data_points}.pkl", 'wb') as file:
				pickle.dump(trg_llc, file)
			self.num_data_points += 1
	
	def len(self):
		return self.num_data_points

	@property
	def raw_paths(self):
		return zip(sorted(self.llc_path.iterdir())[:self.arg.data_point_num], sorted(self.src_path.iterdir())[:self.arg.data_point_num])
	
	@property
	def processed_file_names(self):
		pass # TODO: add cache list here when 100% sure we're done with data handing. leaving this empty so we are forced to redo the entire cache every time we run the model
	
	def load_data(self, src_path, llc_path):
		self.__load_src(src_path)
		self.__load_llc(llc_path)
		print("Dataset size:", len(self.trg_ast), "\nRejected files:", self.arg.data_point_num - len(self.trg_ast))
		return self.trg_ast, self.trg_llc

	def __load_src(self, path):
		for i in tqdm(sorted(path.iterdir())[:self.arg.data_point_num], desc="Loading java source"):
			with open(i) as file:
				try:
					self.trg_ast.append(self.__proc_ast(file))
				except:
					self.exclude.add(i)
    				
	def __load_llc(self, path):
		ii = 0
		for i in tqdm(sorted(path.iterdir())[:self.arg.data_point_num], desc="Loading bytecode"):
			with open(i) as llc_file:
				if i.stem in self.exclude: continue
				try:
					self.trg_llc.append(self.__load_bytecode(llc_file))
				except:
					del self.trg_ast[ii]
					continue
				ii += 1
		pass

	def __proc_ast(self, in_file):
		parsed_src = javalang.parse.parse(in_file.read())
		self.ast = nx.Graph()
		self.ast_idx = 0
		self.ast.add_node(self.ast_idx, type=self.type_map.index(javalang.tree.ClassDeclaration))
		self.__propagate_ast(None, parsed_src.types[0])
		return from_networkx(self.ast, group_node_attrs=['type'])

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
		instruction_identifier = ['aaload', 'aastore', 'aconst_null', 'aload', 'aload_0', 'aload_1', 'aload_2', 'aload_3', 'anewarray', 'areturn', 'arraylength', 'astore', 'astore_0', 'astore_1', 'astore_2', 'astore_3', 'athrow', 'baload', 'bastore', 'bipush', 'breakpoint', 'caload', 'castore', 'checkcast', 'd2f', 'd2i', 'd2l', 'dadd', 'daload', 'dastore', 'dcmpg', 'dcmpl', 'dconst_0', 'dconst_1', 'ddiv', 'dload', 'dload_0', 'dload_1', 'dload_2', 'dload_3', 'dmul', 'dneg', 'drem', 'dreturn', 'dstore', 'dstore_0', 'dstore_1', 'dstore_2', 'dstore_3', 'dsub', 'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2', 'f2d', 'f2i', 'f2l', 'fadd', 'faload', 'fastore', 'fcmpg', 'fcmpl', 'fconst_0', 'fconst_1', 'fconst_2', 'fdiv', 'fload', 'fload_0', 'fload_1', 'fload_2', 'fload_3', 'fmul', 'fneg', 'frem', 'freturn', 'fstore', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3', 'fsub', 'getfield', 'getstatic', 'goto', 'goto_w', 'i2b', 'i2c', 'i2d', 'i2f', 'i2l', 'i2s', 'iadd', 'iaload', 'iand', 'iastore', 'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4', 'iconst_5', 'idiv', 'if_acmpeq', 'if_acmpne', 'if_icmpeq', 'if_icmpge', 'if_icmpgt', 'if_icmple', 'if_icmplt', 'if_icmpne', 'ifeq', 'ifge', 'ifgt', 'ifle', 'iflt', 'ifne', 'ifnonnull', 'ifnull', 'iinc', 'iload', 'iload_0', 'iload_1', 'iload_2', 'iload_3', 'impdep1', 'impdep2', 'imul', 'ineg', 'instanceof', 'invokedynamic', 'invokeinterface', 'invokespecial', 'invokestatic', 'invokevirtual', 'ior', 'irem', 'ireturn', 'ishl', 'ishr', 'istore', 'istore_0', 'istore_1', 'istore_2', 'istore_3', 'isub', 'iushr', 'ixor', 'jsr†', 'jsr_w†', 'l2d', 'l2f', 'l2i', 'ladd', 'laload', 'land', 'lastore', 'lcmp', 'lconst_0', 'lconst_1', 'ldc', 'ldc_w', 'ldc2_w', 'ldiv', 'lload', 'lload_0', 'lload_1', 'lload_2', 'lload_3', 'lmul', 'lneg', 'lookupswitch', 'lor', 'lrem', 'lreturn', 'lshl', 'lshr', 'lstore', 'lstore_0', 'lstore_1', 'lstore_2', 'lstore_3', 'lsub', 'lushr', 'lxor', 'monitorenter', 'monitorexit', 'multianewarray', 'new', 'newarray', 'nop', 'pop', 'pop2', 'putfield', 'putstatic', 'ret†', 'return', 'saload', 'sastore', 'sipush', 'swap', 'tableswitch', 'wide', '(no name)']
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

					k = instruction_identifier.index(instructionName) + 4
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