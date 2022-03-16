import numpy as np
import torch
from utils import NoamOpt

from model.transformer_tree_model import Decoder_AST, Encoder, Graph_NN


class GCN:

	def __init__(self, arg, trg_ast, trg_llc):
		self.arg = arg
		vocab_len = torch.count_nonzero(torch.bincount(torch.cat([i.ndata['n_type'] for i in trg_llc])))
		self.gnn_llc = Graph_NN(annotation_size=vocab_len, out_feats=arg.hid_dim,n_steps=arg.n_gnn_layers,device=arg.device)
		self.gnn_ast = Graph_NN(annotation_size=None,out_feats=arg.hid_dim,n_steps=arg.n_gnn_layers,device=arg.device)
		self.enc = Encoder(vocab_len,arg.hid_dim,arg.n_layers,arg.n_heads,arg.pf_dim,arg.dropout,arg.device,arg.mem_dim,embedding_flag=arg.embedding_flag)
		self.dec = Decoder_AST(arg.output_dim,arg.hid_dim,arg.n_layers,arg.n_heads,arg.pf_dim,arg.dropout,arg.device)
	
	def train(self, trn, tst):
		optim = NoamOpt(self.arg.hid_dim, self.arg.lr_ratio, self.arg.warmup, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
		crit = torch.nn.CrossEntropyLoss()
		for i in trn:
			enc_llc = self.gnn_llc(i)
			enc_ast = self.gnn_ast(i)
			enc_llc = self.enc(enc_llc)
			output = self.dec(enc_llc, enc_ast)