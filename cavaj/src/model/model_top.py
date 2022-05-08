import logging
import numpy as np
from torch import nn
import torch
from model.utils import NoamOpt
import torch_geometric.nn as pyg
from torch_geometric.data import Data
import torch.nn.functional as F

# these following imports should be removed once TODOs are completed
import javalang
import inspect
import sys

import gc

class cavaj(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.enc = encoder(arg)
		self.dec = decoder(arg)
		self.type_map = [i for name,i in inspect.getmembers(sys.modules[javalang.tree.__name__]) if inspect.isclass(i)] # TODO: Copied this over from data_proc so we have a way to know the number of possibilies for the final softmax layer. pass it in a nicer way later
		self.EOS_TOK = len(self.type_map) + 1 # position of EOS token in the final softman layer
		self.new_node_final = pyg.Linear(arg.hid_dim, self.EOS_TOK + 2)
		self.node_sel_final = pyg.SAGEConv(arg.hid_dim, 1) # 1 for probability of selecting give node
		self.device = arg.device
		self.max_len = arg.ast_max_len

	def forward(self, llc):
		llc = llc.to(self.device)
		enc_out = self.enc(llc) # get encode 'memory'
		idx_map = {} # maps idx cuz we're not sure of the order of incoming nodes
		stop = False
		i = 0
		ast = Data(x=torch.Tensor(1,self.EOS_TOK + 2), edge_index=torch.Tensor(2, 0).long()).cpu() # empty ast to built on top off
		ast.x[0,self.EOS_TOK-1] = 1 # add SOS token
		edge_data = []
		# TODO: add embedding to AST
		while not stop and i < self.max_len:
			dec_out = self.dec(ast.clone().detach().to(self.device), enc_out)
			new_node = Data(self.new_node_final(dec_out.x), dec_out.edge_index)
			new_node.x = pyg.global_add_pool(new_node.x, batch=None) # collapse output of variable size to a single 1 # TODO: try different pooling methods
			new_node = F.log_softmax(new_node.x, dim=1) # get class probabilities
			node_sel = Data(self.node_sel_final(dec_out.x, dec_out.edge_index), dec_out.edge_index) # get next node to connect to

			stop = torch.argmax(new_node).item() == self.EOS_TOK # check if End Of Sequence token is the new prediction

			# if torch.argmax(new_node) < self.EOS_TOK-1: # if current node is not SOS or EOS
			edge_data.append(node_sel.x.reshape(-1).cpu())
			ast.edge_index = torch.hstack([ast.edge_index, torch.hstack([torch.Tensor([ast.num_nodes]), torch.argmax(node_sel.x).cpu()]).unsqueeze(0).T.long()]) # add new edge to ast being build
			ast.edge_index = torch.hstack([ast.edge_index, torch.hstack([torch.argmax(node_sel.x).cpu(), torch.Tensor([ast.num_nodes])]).unsqueeze(0).T.long()]) # add reverse edge to create a DiGraph so ther graph is non single directional

			ast.x = torch.cat([ast.x, new_node.cpu()]) # Add new node to ast being build
			i += 1
		
		ast.edge_attr = torch.zeros(len(edge_data), edge_data[-1].shape[0]).cpu() # padded sqaure matrix to fit triangular matrix in
		for i,ii in enumerate(edge_data):
			ast.edge_attr[i,:ii.shape[0]] = ii

		return ast

# biggest things omitted for simplicity are masks, pos encoder, residuals and dropout
# general structure should be unchanged. some layer dims might need tweeks

# TODO: make vocab length fixed according to the number of all possible bytecode instructions so that it would work with any inference
class encoder(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.src_embed = pyg.SAGEConv(-1, arg.hid_dim) # embedding
		self.enc_units = nn.ModuleList([enc_unit(arg.hid_dim, arg.n_heads) for _ in range(arg.encdec_units)]) # encode units that do the actual attention
	
	def forward(self, x):
		x.x = self.src_embed(x.x, x.edge_index)
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
		x.x = torch.argmax(x.x,1).reshape(-1,1).float()
		x.x = self.dec_embed(x.x,x.edge_index)
		for dec in self.dec_units:
			x = dec(x, llc_enc)
		return x

class dec_unit(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.ast_att = attention(dim, n_heads)
		# self.ast_lcc_att = pyg.TransformerConv(dim, dim, n_heads)
		# self.ast_llc_cat = pyg.Linear(dim * n_heads, dim)
		self.ast_lcc_att = nn.MultiheadAttention(dim, n_heads)
		self.norm = pyg.LayerNorm(dim)
		self.feed_for = feed_forward(dim)
	
	def forward(self, ast, llc_enc):
		ret = self.ast_att(ast)
		# ret.x = self.ast_lcc_att((llc_enc.x,ret.x), ret.edge_index)
		ret.x,_ = self.ast_lcc_att(ret.x, llc_enc.x, llc_enc.x)
		# ret.x = self.ast_llc_cat(ret.x)
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
		x.x = self.att(x.x, x.edge_index)
		x.x = self.concat(x.x)
		x.x = self.norm(x.x)
		return x

class feed_forward(nn.Module):

	def __init__(self, dim) -> None:
		super().__init__()
		self.propagate = pyg.SAGEConv(dim, dim)
		self.norm = pyg.LayerNorm(dim)
	
	def forward(self, x):
		x.x = self.propagate(x.x, x.edge_index)
		x.x = self.norm(x.x)
		return x