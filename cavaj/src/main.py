from pathlib import Path

import logging
from tqdm import tqdm
from datetime import datetime
from contextlib import nullcontext

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

import param
from data_proc import data_proc
from model.model_top import cavaj
from model.utils import NoamOpt
from torch_geometric.utils.convert import from_networkx, to_networkx


def checkpoint_model(epoch, model, optim, checkpoint_path):
	torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, checkpoint_path)


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG, filename="./log/cavaj.log", force=True, filemode='w')
	log_time = datetime.now().replace(microsecond=0)

	# Making results reproducible for debugging. TODO: remove when done
	torch.manual_seed(0)
	# torch.use_deterministic_algorithms(True, warn_only=True)

	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	data = data_proc(arg, dataset_path / "java_src", dataset_path / "bytecode", dataset_path / "cache")
	checkpoint_path = Path("../model_checkpoints/checkpoint.pt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	arg.device = device
	tb = SummaryWriter(f'./log/tb/run_{log_time.isoformat()}')

	model = cavaj(arg).to(device)

	train = data[:int(len(data) * 0.7)]
	val = data[int(len(data) * 0.7):]

	# optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	optim = torch.optim.Adam(model.parameters(), lr=arg.lr_rate, weight_decay=5e-4)
	crit = torch.nn.CrossEntropyLoss()

	crit = nn.CrossEntropyLoss()
	epoch_iter = tqdm(range(arg.epochs), desc="Epochs", disable=arg.no_prog)
	for i in epoch_iter:
		trn_iter = tqdm(enumerate(train), total=len(train), disable=arg.no_prog)
		failed = 0
		tot_acc = 0
		tot_loss = 0
		with profile(schedule=schedule(wait=2, warmup=2, active=6), on_trace_ready=tensorboard_trace_handler(f'./log/tb/profiler_{log_time.isoformat()}'), profile_memory=True) if arg.profile else nullcontext() as prof:
			for j,batch in trn_iter:
				# batch[0][1] = batch[0][1].to(device)
				try:
					optim.zero_grad()
					out = model(batch[1])
					out.x = out.x[1:]
					new_predicted = to_networkx(out)
					new_truth = to_networkx(batch[0][1])
					ged_gen = nx.optimize_graph_edit_distance(new_truth, new_predicted)
					node_loss = crit(out.x[:batch[0][1].x.shape[0]], batch[0][1].x[:out.x.shape[0]].T.squeeze(0))
					edge_loss = crit(out.edge_attr[:batch[0][0].shape[0]], batch[0][0][:out.edge_attr.shape[0],1])
					loss = (node_loss + edge_loss)/2
					loss.backward()
					optim.step()
					if arg.profile:
						prof.step()
					ged = next(ged_gen)
					acc = 1 - (ged / (max(len(new_predicted.nodes)+len(new_predicted.edges),len(new_truth.nodes)+len(new_truth.edges))))
					tot_acc += acc
					tot_loss += loss.item()
					node_loss = node_loss.item()
					edge_loss = edge_loss.item()
				except Exception as e:
					failed += 1
					logging.warning(f"failed (total {failed}) to train {j} with exception {e}")
					if failed > len(train) * 0.5: # throw error if most data is rejected
						logging.error(f"failed to train on more than 50% of dataset ({failed} batches failed)")
					raise
				except KeyboardInterrupt:
					checkpoint_model(i, model, optim, checkpoint_path)
				trn_iter.set_description(f"Node Loss: {node_loss:.3f}, Edge Loss: {edge_loss:.3f}")
				tb.add_scalar("Training Loss", loss, j)
				tb.add_scalar("Training Acc", acc, j)
				logging.info(f"Epoch Train: {i:3d}, Element: {j:3d} Node Loss: {node_loss:7.2f}, Edge Loss: {edge_loss:7.2f}, Acc: {acc:7.2f}, Graph size: {out.x.shape[0]}")
			if i % arg.chk_interval:
				checkpoint_model(i, model, optim, checkpoint_path)
		trn_acc = tot_acc / (j - failed)
		trn_loss = tot_loss / (j - failed)
		logging.INFO(f"Epoch Train {i:3d}, Average Acc: {trn_acc:7.2f}, Average Loss: {trn_loss:7.2f}")
		tb.add_scalar("Average Training Loss", trn_loss, i)
		tb.add_scalar("Average Training Acc", trn_acc, i)

		failed = 0
		tot_acc = 0
		tot_loss = 0
		for j,batch in trn_iter:
			# batch[0][1] = batch[0][1].to(device)
			try:
				with torch.no_grad():
					out = model(batch[1])
					out.x = out.x[1:]
					new_predicted = to_networkx(out)
					new_truth = to_networkx(batch[0][1])
					ged_gen = nx.optimize_graph_edit_distance(new_truth, new_predicted)
					node_loss = crit(out.x[:batch[0][1].x.shape[0]], batch[0][1].x[:out.x.shape[0]].T.squeeze(0))
					edge_loss = crit(out.edge_attr[:batch[0][0].shape[0]], batch[0][0][:out.edge_attr.shape[0],1])
				loss = (node_loss + edge_loss)/2
				ged = next(ged_gen)
				acc = 1 - (ged / (max(len(new_predicted.nodes)+len(new_predicted.edges),len(new_truth.nodes)+len(new_truth.edges))))
				tot_acc += acc
				tot_loss += loss.item()
				node_loss = node_loss.item()
				edge_loss = edge_loss.item()
			except Exception as e:
				failed += 1
				logging.warning(f"failed (total {failed}) to validate {j} with exception {e}")
				if failed > len(train) * 0.5: # throw error if most data is rejected
					logging.error(f"failed to validate on more than 50% of dataset ({failed} batches failed)")
				raise
			except KeyboardInterrupt:
				checkpoint_model(i, model, optim, checkpoint_path)
			trn_iter.set_description(f"Node Loss: {node_loss:.3f}, Edge Loss: {edge_loss:.3f}")
			tb.add_scalar("Validation Loss", loss, j)
			tb.add_scalar("Validation Acc", acc, j)
			logging.info(f"Epoch Validation: {i:3d}, Element: {j:3d} Node Loss: {node_loss:7.2f}, Edge Loss: {edge_loss:7.2f}, Acc: {acc:7.2f}, Graph size: {out.x.shape[0]}")
		val_acc = tot_acc / (j - failed)
		val_loss = tot_loss / (j - failed)
		logging.INFO(f"Epoch Validation {i:3d}, Average Acc: {trn_acc:7.2f}, Average Loss: {trn_loss:7.2f}")
		tb.add_scalar("Average Validation Loss", val_loss, i)
		tb.add_scalar("Average Validation Acc", val_acc, i)
		epoch_iter.set_description(f"Train Loss: {trn_loss}, Acc: {trn_acc}. Validation Loss: {val_loss}, Acc: {val_acc}")
	tb.close()
