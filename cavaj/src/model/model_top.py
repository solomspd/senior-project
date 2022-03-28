import numpy as np
from torch import nn
from model.utils import NoamOpt
import torch.nn.functional as F
import torch_geometric.nn as pyg
from torch_geometric.data import Data

# these following imports should be removed once TODOs are completed
import networkx as nx
import javalang
import inspect
import sys

# from model.transformer_tree_model import Decoder_AST, Encoder, Graph_NN

class cavaj(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.enc = encoder(arg)
		self.dec = decoder(arg)
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] # TODO: Copied this over from data_proc so we have a way to know the number of possibilies for the final softmax layer. pass it in a nicer way later
		self.EOS_TOK = arg.out_dim + 1 # position of EOS token in the final softman layer
		self.new_node_final = pyg.Linear(arg.hid_dim, self.EOS_TOK)
		self.node_sel_final = pyg.SAGEConv(arg.hid_dim, 1) # 1 for probability of selecting give node

	def forward(self, ast, llc):
		enc_out = self.enc(llc)
		final_out = nx.Graph()
		stop = False
		i = 0
		while not stop:
			dec_out = self.dec(ast, enc_out)
			new_node = Data(self.nw_final(dec_out.x, dec_out.edge_index), dec_out.edge_index)
			new_node = F.log_softmax(new_node.x)
			node_sel = Data(self.node_sel_final(dec_out.x, dec_out.edge_index), dec_out.edge_index)
			stop = new_node == self.EOS_TOK # check if End Of Sequence token is the new prediction

			# apparently pytorch can accumulate loss like this. pretty neat
			F.nll_loss(new_node, node_ground_truth[i]).backward()
			F.mse_loss(node_sel, node_ground_truth[i]).backward()

			# FIXME: realized this is a bad way of going about it. use pytorch geoemtric instead of networkx for final_out
			final_out.add_node(new_node.x) # TODO: CHANGE THIS TO WORK WITH BATCHS INSTEAD OF INDIVIDUAL NODES BEFORE USING BATCHINGEFORE USING BATCHING
			final_out.add_edge()

		return final_out

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
		x = Data(self.dec_embed(x.x,x.edge_index), x.edge_index)
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
		ret = Data(self.ast_lcc_att((ret.x,llc_enc.x), ret.edge_index), ret.edge_index)
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