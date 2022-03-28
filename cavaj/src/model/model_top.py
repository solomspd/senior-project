import numpy as np
from torch import nn
import torch
from model.utils import NoamOpt
import torch.nn.functional as F
import torch_geometric.nn as pyg
from torch_geometric.data import Data

# these following imports should be removed once TODOs are completed
import javalang
import inspect
import sys

class cavaj(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.enc = encoder(arg)
		self.dec = decoder(arg)
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] # TODO: Copied this over from data_proc so we have a way to know the number of possibilies for the final softmax layer. pass it in a nicer way later
		self.EOS_TOK = len(self.type_map) # position of EOS token in the final softman layer
		self.new_node_final = pyg.Linear(arg.hid_dim, self.EOS_TOK + 1)
		self.node_sel_final = pyg.SAGEConv(arg.hid_dim, 1) # 1 for probability of selecting give node
		self.device = arg.device

	def forward(self, ground_truth, llc):
		ground_truth = ground_truth.to(self.device)
		llc = llc.to(self.device)
		enc_out = self.enc(llc)
		idx_map = {} # maps idx cuz we're not sure of the order of incoming nodes
		stop = False
		i = 0
		ast = Data(x=torch.Tensor([[1]]), edge_index=torch.Tensor(2, 0).long()).to(self.device)
		loss_avg = 0
		while not stop and i < len(ground_truth):
			dec_out = self.dec(ast, enc_out)
			new_node = Data(self.new_node_final(dec_out.x), dec_out.edge_index)
			new_node.x = pyg.global_add_pool(new_node.x, batch=None)
			new_node = F.log_softmax(new_node.x, dim=1) # get class probabilities
			node_sel = Data(self.node_sel_final(dec_out.x, dec_out.edge_index), dec_out.edge_index) # get next node to connect to

			stop = torch.argmax(new_node).item() == self.EOS_TOK # check if End Of Sequence token is the new prediction

			# apparently pytorch can accumulate loss like this. pretty neat
			idx_map[ground_truth[i][2].item()] = i
			loss = F.nll_loss(new_node, ground_truth[i][0].unsqueeze(0)) # back prop new node type
			new_node = torch.argmax(new_node, dim=1) # get the class that was predicted
			if ground_truth[i][1] >= 0:
				sel_graph_truth = torch.zeros(node_sel.x.shape).to(self.device) # Turn the current index to a graph tensor to compare to
				sel_graph_truth[idx_map[ground_truth[i][1].item()]] = 1
				loss += F.mse_loss(node_sel.x, sel_graph_truth) # back prop new edge
				ast.edge_index = torch.hstack([ast.edge_index, torch.hstack([torch.Tensor([ast.num_nodes]).to(self.device), torch.argmax(node_sel.x)]).unsqueeze(0).T.long()]) # Add new edge to ast being build
			
			loss_avg += loss.item()

			loss.backward(retain_graph=True)

			# TODO: CHANGE THIS TO WORK WITH BATCHS INSTEAD OF INDIVIDUAL NODES BEFORE USING BATCHINGEFORE USING BATCHING
			ast.x = torch.cat([ast.x, new_node.unsqueeze(0)]) # Add new node to ast being build
			i += 1

		return ast, loss_avg/i

# biggest things omitted for simplicity are masks, pos encoder, residuals and dropout
# general structure should be unchanged. some layer dims might need tweeks
# only real remaining quesiton is the final classifier

# TODO: make vocab length fixed according to the number of all possible bytecode instructions so that it would work with any inference
class encoder(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.src_embed = pyg.SAGEConv(-1, arg.hid_dim) # embedding
		self.enc_units = nn.ModuleList([enc_unit(arg.hid_dim, arg.n_heads) for _ in range(arg.encdec_units)]) # encode units that do the actual attention
	
	def forward(self, x):
		x = Data(self.src_embed(x.x, x.edge_index), x.edge_index)
		for enc in self.enc_units:
			x = enc(x)
		return x

class enc_unit(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.att = attention(dim, n_heads)
		self.feed_for = feed_forward(dim)
	
	def forward(self, x):
		x = self.att(x)
		x = self.feed_for(x)
		return x


class decoder(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.dec_embed = pyg.SAGEConv(-1, arg.hid_dim)
		self.dec_units = nn.ModuleList([dec_unit(arg.hid_dim, arg.hid_dim) for _ in range(arg.encdec_units)])
	
	def forward(self, x, llc_enc):
		embed = self.dec_embed(x.x,x.edge_index)
		x = Data(embed, x.edge_index)
		for dec in self.dec_units:
			x = dec(x, llc_enc)
		return x

class dec_unit(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.ast_att = attention(dim, n_heads)
		self.ast_lcc_att = pyg.TransformerConv(dim, dim, n_heads)
		self.ast_llc_cat = pyg.Linear(dim * n_heads, dim)
		self.norm = pyg.LayerNorm(dim)
		self.feed_for = feed_forward(dim)
	
	def forward(self, ast, llc_enc):
		ret = self.ast_att(ast)
		ret = Data(self.ast_lcc_att((llc_enc.x,ret.x), ret.edge_index), ret.edge_index)
		ret.x = self.ast_llc_cat(ret.x)
		ret.x = self.norm(ret.x)
		ret = self.feed_for(ret)
		return ret

class attention(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.att = pyg.TransformerConv(dim, dim, n_heads)
		self.concat = pyg.Linear(dim * n_heads, dim) # TODO: maybe try smth besides linear idk
		self.norm = pyg.LayerNorm(dim)
	
	def forward(self, x):
		x = Data(self.att(x.x, x.edge_index), x.edge_index)
		x.x = self.concat(x.x)
		x.x = self.norm(x.x)
		return x

class feed_forward(nn.Module):

	def __init__(self, dim) -> None:
		super().__init__()
		self.propagate = pyg.SAGEConv(dim, dim)
		self.norm = pyg.LayerNorm(dim)
	
	def forward(self, x):
		x = Data(self.propagate(x.x, x.edge_index), x.edge_index)
		x.x = self.norm(x.x)
		return x