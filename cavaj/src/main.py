from pathlib import Path

import logging
from tqdm import tqdm
from datetime import datetime
from contextlib import nullcontext
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
from scipy.spatial import distance

import param

from data_proc import data_proc
from model.model_top import cavaj
from model.utils import NoamOpt
from torch_geometric.utils.convert import from_networkx, to_networkx


def checkpoint_model(epoch, model, optim, checkpoint_path, arg):
	torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(), 'args': {'device': arg.device, 'hid_dim': arg.hid_dim, 'ast_max_len': arg.ast_max_len, 'n_heads': arg.n_heads, 'encdec_units': arg.encdec_units}}, checkpoint_path / f"checkpoint-epoch {epoch}-{datetime.now().isoformat()}")

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG, filename="./log/cavaj.log", force=True, filemode='w')
	log_time = datetime.now().replace(microsecond=0)

	# Making results reproducible for debugging. TODO: remove when done
	torch.manual_seed(0)
	# torch.use_deterministic_algorithms(True, warn_only=True)

	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	data = data_proc(arg, dataset_path / "java_src", dataset_path / "bytecode", dataset_path / "cache")
	checkpoint_path = Path(f"../model_checkpoints/{log_time.isoformat()}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	arg.device = device
	tb = SummaryWriter(f'./log/tb/run_{log_time.isoformat()}')


	model = cavaj(arg).to(device)

	# optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	optim = torch.optim.Adam(model.parameters(), lr=arg.lr_rate, weight_decay=5e-4)
	crit = torch.nn.CrossEntropyLoss()
	epoch = 0
	checkpoint_path.mkdir()
	if arg.checkpoint is not None:
		checkpoint = torch.load(arg.checkpoint, map_location='cpu')
		model.load_state_dict(checkpoint['model_state_dict'])
		optim.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']

	train = data[:int(len(data) * 0.7)]
	val = data[int(len(data) * 0.7):]

	epoch_iter = tqdm(range(epoch, arg.epochs), desc="Epochs", initial=epoch, total=arg.epochs, disable=arg.no_prog)
	for i in epoch_iter:
		trn_iter = tqdm(enumerate(train), total=len(train), disable=arg.no_prog)
		failed = 0
		tot_ged_acc = 0
		tot_ele_node_acc = 0
		tot_ele_edge_acc = 0
		tot_node_cos = 0
		tot_edge_cos = 0
		tot_loss = 0
		with profile(schedule=schedule(wait=2, warmup=2, active=6), on_trace_ready=tensorboard_trace_handler(f'./log/tb/profiler_{log_time.isoformat()}'), profile_memory=True) if arg.profile else nullcontext() as prof:
			for j,batch in trn_iter:
				# batch[0][1] = batch[0][1].to(device)
				try:
					optim.zero_grad()
					out = model(batch[1])
					out.x = out.x[1:]
					node_pred = out.x[:batch[0][1].x.shape[0]]
					node_truth = batch[0][1].x[:out.x.shape[0]].T.squeeze(0)
					edge_pred = out.edge_attr[:batch[0][0].shape[0]]
					edge_truth = batch[0][0][:out.edge_attr.shape[0],1]
					node_loss = crit(node_pred, node_truth)
					edge_loss = crit(edge_pred, edge_truth)
					node_pred = torch.argmax(node_pred, axis=1)
					edge_pred = torch.argmax(edge_pred, axis=1)
					new_predicted = to_networkx(out)
					new_truth = to_networkx(batch[0][1])
					ged_gen = nx.optimize_graph_edit_distance(new_truth, new_predicted)
					ele_node_acc = node_pred == node_truth
					ele_edge_acc = edge_pred == edge_truth
					bleu_node = ele_node_acc
					bleu_edge = ele_edge_acc
					node_cos = 1 - distance.cosine(node_pred.numpy(), node_truth.numpy())
					edge_cos = 1 - distance.cosine((edge_pred+1).numpy(), (edge_truth+1).numpy())
					ele_node_acc = torch.sum(ele_node_acc)/len(ele_node_acc)
					ele_edge_acc = torch.sum(ele_edge_acc)/len(ele_edge_acc)
					loss = (node_loss + edge_loss)/2
					loss.backward()
					loss = loss.item()
					optim.step()
					if arg.profile:
						prof.step()
					ged = next(ged_gen)
					ged_acc = 1 - (ged / (max(len(new_predicted.nodes)+len(new_predicted.edges),len(new_truth.nodes)+len(new_truth.edges))))
					tot_ged_acc += ged_acc
					tot_ele_node_acc += ele_node_acc
					tot_ele_edge_acc += ele_edge_acc
					tot_node_cos += node_cos
					tot_edge_cos += edge_cos
					tot_loss += loss
					node_loss = node_loss.item()
					edge_loss = edge_loss.item()
				except Exception as e:
					failed += 1
					logging.warning(f"failed (total {failed}) to train {j} with exception {e}")
					checkpoint_model(i, model, optim, checkpoint_path, arg)
					if failed > len(train) * 0.5: # throw error if most data is rejected
						logging.error(f"failed to train on more than 50% of dataset ({failed} batches failed)")
					raise
				except KeyboardInterrupt:
					checkpoint_model(i, model, optim, checkpoint_path, arg)
				trn_iter.set_description(f"Training Node Loss: {node_loss:.3f}, Edge Loss: {edge_loss:.3f}")
				tb.add_scalar("Atomic Training/Loss", loss, j)
				tb.add_scalar("Atomic Training/GED Accuracy", ged_acc, j)
				tb.add_scalar("Atomic Training/element wise node Accuracy", ele_node_acc, j)
				tb.add_scalar("Atomic Training/element wise edge Accuracy", ele_edge_acc, j)
				tb.add_scalar("Atomic Training/Cosine node similarity", node_cos, j)
				tb.add_scalar("Atomic Training/Cosine edge similarity", edge_cos, j)
				logging.info(f"Epoch Train: {i:3d}, Element: {j:3d} Node Loss: {node_loss:7.2f}, Edge Loss: {edge_loss:7.2f}, Node Acc: {ele_node_acc:7.2f}, Edge Acc: {ele_edge_acc:7.2f}, Cosine node: {node_cos:.7f}, Cosine edge: {edge_cos:.7f}, GED Acc: {ged_acc:7.2f}, Graph size: {out.x.shape[0]}")

		checkpoint_model(i, model, optim, checkpoint_path, arg)

		trn_ged_acc = tot_ged_acc / (j - failed)
		trn_ele_node_acc = tot_ele_node_acc / (j - failed)
		trn_ele_edge_acc = tot_ele_edge_acc / (j - failed)
		trn_node_cos = tot_node_cos / (j - failed)
		trn_edge_cos = tot_edge_cos / (j - failed)
		trn_loss = tot_loss / (j - failed)
		logging.info(f"Epoch Train: {i:3d}, Average GED Acc: {trn_ged_acc:7.2f}, Average Loss: {trn_loss:7.2f}")
		tb.add_scalar("Training/Loss", trn_loss, i)
		tb.add_scalar("Training/GED Accuracy", trn_ged_acc, i)
		tb.add_scalar("Training/Element wise node Accuracy", trn_ele_node_acc, i)
		tb.add_scalar("Training/Element wise edge Accuracy", trn_ele_edge_acc, i)
		tb.add_scalar("Training/Cosine node similarity", trn_node_cos, j)
		tb.add_scalar("Training/Cosine edge similarity", trn_edge_cos, j)

		failed = 0
		tot_ged_acc = 0
		tot_ele_node_acc = 0
		tot_ele_edge_acc = 0
		tot_node_cos = 0
		tot_edge_cos = 0
		tot_loss = 0
		val_iter = tqdm(enumerate(val), total=len(val), disable=arg.no_prog)
		for j,batch in val_iter:
			# batch[0][1] = batch[0][1].to(device)
			try:
				with torch.no_grad():
					out = model(batch[1])
					out.x = out.x[1:]
					node_pred = out.x[:batch[0][1].x.shape[0]]
					node_truth = batch[0][1].x[:out.x.shape[0]].T.squeeze(0)
					edge_pred = out.edge_attr[:batch[0][0].shape[0]]
					edge_truth = batch[0][0][:out.edge_attr.shape[0],1]
					node_loss = crit(node_pred, node_truth)
					edge_loss = crit(edge_pred, edge_truth)
					node_pred = torch.argmax(node_pred, axis=1)
					edge_pred = torch.argmax(edge_pred, axis=1)
					new_predicted = to_networkx(out)
					new_truth = to_networkx(batch[0][1])

					# nx.draw(new_predicted, with_labels= True, node_size = 7, font_size = 0.05)			
					# plt.savefig('predicted hierarchy.png', dpi = 1000)

					# nx.draw(new_truth, with_labels= True, node_size = 7, font_size = 0.05)
					# plt.savefig('truth hierarchy.png', dpi = 1000)

					ged_gen = nx.optimize_graph_edit_distance(new_truth, new_predicted)
					node_pred, edge_pred = node_pred.float(), edge_pred.float()
					node_cos = 1 - distance.cosine(node_pred.numpy(), node_truth.numpy())
					edge_cos = 1 - distance.cosine((edge_pred+1).numpy(), (edge_truth+1).numpy())
					ele_node_acc = node_pred == node_truth
					ele_edge_acc = edge_pred == edge_truth
					ele_node_acc = torch.sum(ele_node_acc)/len(ele_node_acc)
					ele_edge_acc = torch.sum(ele_edge_acc)/len(ele_edge_acc)
				loss = (node_loss + edge_loss)/2
				loss = loss.item()
				ged = next(ged_gen)
				ged_acc = 1 - (ged / (max(len(new_predicted.nodes)+len(new_predicted.edges),len(new_truth.nodes)+len(new_truth.edges))))
				tot_ged_acc += ged_acc
				tot_ele_node_acc += ele_node_acc
				tot_ele_edge_acc += ele_edge_acc
				tot_node_cos += node_cos
				tot_edge_cos += edge_cos
				tot_loss += loss
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
			val_iter.set_description(f"Validation Node Loss: {node_loss:.3f}, Edge Loss: {edge_loss:.3f}, Acc: {ged_acc:7.2f}")
			tb.add_scalar("Atomic Validation/Loss", loss, j)
			tb.add_scalar("Atomic Validation/GED Accuracy", ged_acc, j)
			tb.add_scalar("Atomic Validation/Element wise node Accuracy", ele_node_acc, j)
			tb.add_scalar("Atomic Validation/Element wise edge Accuracy", ele_edge_acc, j)
			tb.add_scalar("Atomic Validation/Cosine node similarity", node_cos, j)
			tb.add_scalar("Atomic Validation/Cosine edge similarity", edge_cos, j)
			logging.info(f"Epoch Validation: {i:3d}, Element: {j:3d} Node Loss: {node_loss:7.2f}, Edge Loss: {edge_loss:7.2f}, Node Acc: {ele_node_acc:7.2f}, Edge Acc: {ele_edge_acc:7.2f}, Cosine node: {node_cos:.7f}, Cosine edge: {edge_cos:.7f}, GED Acc: {ged_acc:7.2f}, Graph size: {out.x.shape[0]}")
		val_ged_acc = tot_ged_acc / (j - failed)
		val_ele_node_acc = tot_ele_node_acc / (j - failed)
		val_ele_edge_acc = tot_ele_edge_acc / (j - failed)
		val_node_cos = tot_node_cos / (j - failed)
		val_edge_cos = tot_edge_cos / (j - failed)
		val_loss = tot_loss / (j - failed)
		logging.info(f"Epoch Validation: {i:3d}, Average GED Acc: {val_ged_acc:7.2f}, Average Loss: {val_loss:7.2f}")
		tb.add_scalar("Validation/Loss", val_loss, i)
		tb.add_scalar("Validation/GED Accuracy", val_ged_acc, i)
		tb.add_scalar("Validation/Element wise node Accuracy", val_ele_node_acc, i)
		tb.add_scalar("Validation/Element wise edge Accuracy", val_ele_edge_acc, i)
		tb.add_scalar("Validation/Cosine node similarity", val_node_cos, i)
		tb.add_scalar("Validation/Cosine edge similarity", val_edge_cos, i)
		epoch_iter.set_description(f"Train Loss: {trn_loss:7.2f}, Acc: {trn_ged_acc:7.2f}. Validation Loss: {val_loss:7.2f}, Acc: {val_ged_acc:7.2f}")

	tb.close()
