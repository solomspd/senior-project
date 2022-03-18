from turtle import forward
import numpy as np
from torch import nn
from model.utils import NoamOpt
import torch
import torch_geometric.nn as pyg

# from model.transformer_tree_model import Decoder_AST, Encoder, Graph_NN

class cavaj(nn.Module):

	def __init__(self, arg, trg_ast, trg_llc) -> None:
		self.enc = encoder(arg)
		self.dec = decoder(arg)

	def forward(self, ast, llc):
		enc_out = self.enc(llc)
		dec_out = self.dec(ast, enc_out)
		return dec_out


# biggest things omitted for simplicity are masks, pos encoder, residuals and dropout
# general structure should be unchanged. some layer dims might need tweeks
# only real remaining quesiton is the final classifier

			
class encoder(nn.Module):

	def __init__(self, arg) -> None:
		super().__init__()
		self.src_embed = pyg.SAGEConv(arg.hid_dim, arg.hid_dim) # embedding
		self.enc_units = nn.ModuleList([enc_unit(arg.hid_dim, arg.hid_dim, arg.n_heads) for _ in range(arg.encdec_units)]) # encode units that do the actual attention
	
	def forward(self, x):
		x = self.src_embed(x)
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
		self.dec_embed = pyg.SAGEConv(arg.hid_dim, arg.hid_dim)
		self.dec_units = nn.ModuleList([dec_unit(arg.hid_dim, arg.hid_dim) for _ in range(arg.encdec_units)])
	
	def attention(self, x):
		x = self.dec_embed(x)
		for dec in self.dec_units:
			x = dec(x)
		return x

class dec_unit(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.ast_att = attention(dim, n_heads)
		self.ast_lcc_att = # having issues with this one. simply piping the graph throught the GAT uses the GATs internal Q,K,V. while in this case we want to use the Q of the ast but he K and V of the llc. if im getting this correctly.
		self.norm = pyg.LayerNorm(dim)
		self.feed_for = feed_forward(dim)
	
	def forward(self, ast, llc_enc):
		ret = self.ast_att(ast)
		ret = self.norm1(ret)
		ret = self.ast_lcc_att(ret, llc_enc)
		ret = self.norm(ret)
		ret = self.feed_for(ret)
		return ret

class attention(nn.Module):

	def __init__(self, dim, n_heads) -> None:
		super().__init__()
		self.att = pyg.GATConv(dim, dim, n_heads)
		self.norm = pyg.LayerNorm(dim)
	
	def forward(self, x):
		x = self.att(x)
		x = self.norm(x)
		return x

class feed_forward(nn.Module):

	def __init__(self, dim) -> None:
		super().__init__()
		self.propagate = pyg.SAGEConv(dim, dim)
		self.norm = pyg.LayerNorm(dim)
	
	def forward(self, x):
		x = self.propagate(x)
		x = self.norm(x)
		return x