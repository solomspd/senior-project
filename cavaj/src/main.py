from pathlib import Path

import logging

import networkx as nx
import torch
from torch_geometric.loader import DataLoader

import param
from data_proc import data_proc
from model.c_dataset import dataset
from model.model_top import cavaj
from model.utils import NoamOpt


if __name__ == '__main__':
	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	data = data_proc(arg, dataset_path / "java_src", dataset_path / "bytecode", dataset_path / "cache")

	logging.basicConfig(level=logging.DEBUG)

	model = cavaj(arg)

	loader = DataLoader(data, batch_size=arg.batch_sz, shuffle=False)

	# moved training to main
	# optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	crit = torch.nn.CrossEntropyLoss()

	model(data[0], data[0])

	for i in arg.epochs:
		for data in loader:
			pass