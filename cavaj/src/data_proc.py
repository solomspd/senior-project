from tqdm import tqdm
from pathlib import Path
import networkx as nx
import dgl

class data_proc:

	def __init__(self):
		self.exclude = set()
		self.trg = []

	def load_src(self, path):
		for i in tqdm(sorted(path.iterdir())):
			with open(i) as file:
				try:
					self.trg.append(self.proc_ast(file))
				except:
					self.exclude.add(i)

	def load_llc(self):
		pass

	def __proc_ast(in_file):
		pass