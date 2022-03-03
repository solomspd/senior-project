
from transformer_tree_model import Graph_NN, Encoder, Decoder_AST


class GCN:

	def __init__(self, arg, trg_ast, trg_llc):
		self.arg = arg
		self.gnn_llc = Graph_NN(annotation_size=None, out_feats=arg.hid_dim,n_steps=arg.n_gnn_layers,device=arg.device)
		self.gnn_ast = Graph_NN(annotation_size=None,out_feats=arg.hid_dim,n_steps=arg.n_gnn_layers,device=arg.device)
		self.enc = Encoder(None,arg.hid_dim,arg.n_layers,arg.n_heads,arg.pf_dim,arg.dropout,arg.device,arg.mem_dim,embedding_flag=arg.embedding_flag,max_length=0)
		self.dec = Decoder_AST(arg.output_dim,arg.hid_dim,arg.n_layers,arg.n_heads,arg.pf_dim,arg.dropout,arg.device,max_length=0)