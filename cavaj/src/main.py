import torch
import dgl
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

from data_proc import data_proc
import param

if __name__ == '__main__':
	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	trg_ast,src_f,src_g,trg_llc = data_proc(arg).load_data(dataset_path / "java_src", dataset_path / "bytecode")
	plt.figure()
	nx.draw(dgl.to_networkx(trg_ast[0]))
	plt.show()