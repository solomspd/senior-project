from pathlib import Path

import logging
from tqdm import tqdm

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
	checkpoint_path = Path("../model_checkpoints/checkpoint.pt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	arg.device = device

	logging.basicConfig(level=logging.DEBUG, filename="cavaj.log", force=True)

	model = cavaj(arg).to(device)

	train = DataLoader(data[:int(len(data) * 0.7)])
	val = DataLoader(data[int(len(data) * 0.7):])
	# loader = DataLoader(data, batch_size=arg.batch_sz, shuffle=False)

	# moved training to main
	optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	crit = torch.nn.CrossEntropyLoss()

	for i in tqdm(range(arg.epochs)):
		btch_iter = tqdm(enumerate(train), total=len(train))
		for j,batch in btch_iter:
			optim.optimizer.zero_grad()
			out,loss = model(batch[0].squeeze(), batch[1])
			optim.step()
			btch_iter.set_description(f"Last Loss {loss:.3f}")
			logging.info(f"Epoch: {i}, element: {j} Loss: {loss}")
		if i % 20:
			torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.optimizer.state_dict()}, checkpoint_path)
