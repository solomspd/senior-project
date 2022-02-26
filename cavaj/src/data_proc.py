from tqdm import tqdm
from pathlib import Path
import networkx as nx
import dgl
import javalang
import sys,inspect

class data_proc:

	def __init__(self):
		self.exclude = set()
		self.trg = []
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)]
		self.ast = nx.graph()

	def load_src(self, path):
		for i in tqdm(sorted(path.iterdir())):
			with open(i) as file:
				try:
					self.trg.append(self.proc_ast(file))
				except:
					self.exclude.add(i)

	def load_llc(self):
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