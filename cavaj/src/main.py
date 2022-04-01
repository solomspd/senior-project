from pathlib import Path

import logging
from tqdm import tqdm

import networkx as nx
import torch
from torch_geometric.loader import DataLoader

import param
from data_proc import data_proc
from model.model_top import cavaj
from model.utils import NoamOpt

def checkpoint_model(epoch, model, optim, checkpoint_path):
	torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, checkpoint_path)


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG, filename="cavaj.log", force=True, filemode='w')
	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	data = data_proc(arg, dataset_path / "java_src", dataset_path / "bytecode", dataset_path / "cache")
	checkpoint_path = Path("../model_checkpoints/checkpoint.pt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	arg.device = device

	model = cavaj(arg).to(device)

	train = DataLoader(data[:int(len(data) * 0.7)])
	val = DataLoader(data[int(len(data) * 0.7):])
	# loader = DataLoader(data, batch_size=arg.batch_sz, shuffle=False)

	# moved training to main
	# optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	optim = torch.optim.Adam(model.parameters(), lr=arg.lr_rate, weight_decay=5e-4)
	crit = torch.nn.CrossEntropyLoss()

	# Making results reproducible for debugging. TODO: remove when done
	torch.manual_seed(0)
	torch.use_deterministic_algorithms(True, warn_only=True)

	failed = 0
	for i in tqdm(range(arg.epochs), desc="Epochs"):
		btch_iter = tqdm(enumerate(train), total=len(train))
		for j,batch in btch_iter:
			optim.zero_grad()
			try:
				out,loss = model(batch[0].squeeze(), batch[1], optim)
			except Exception as e:
				failed += 1
				logging.warning(f"failed (total {failed}) to propagate batch {j} with exception {e}")
				if failed > len(train) * 0.5: # throw error if most data is rejected
					logging.error(f"failed to propagate more than 50% of dataset ({failed} batches failed)")
			except KeyboardInterrupt:
				checkpoint_model(i, model, optim, checkpoint_path)
			btch_iter.set_description(f"Last Loss {loss:.3f}")
			logging.info(f"Epoch: {i}, element: {j} Loss: {loss}")
		if i % arg.chk_interval:
			checkpoint_model(i, model, optim, checkpoint_path)
