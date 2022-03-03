import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Welcome to the Cavaj CLI decompilation tool')
	parser.add_argument("--data_point_num", help="the number of files to be used from the dataset", type=int, default=None)
	parser.add_argument('--hid_dim', type=int, default=256)
	parser.add_argument('--mem_dim', type=int, default=64)
	parser.add_argument('--n_layers', type=int, default=2) 
	parser.add_argument('--n_gnn_layers', type=int, default=2) 
	parser.add_argument('--n_heads', type=int, default=4)
	parser.add_argument('--pf_dim', type=int, default=512)
	parser.add_argument('--max_tolerate_len', type=int, default=1200) #for small GPU mem 
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--depth_dim', type=int, default=40)
	parser.add_argument('--output_dim', type=int, default=128)
	parser.add_argument('--graph_aug', action='store_true', default=True)
	parser.add_argument('--embedding_flag',type=int, default=1)
	parser.add_argument('--lr_ratio', type=float, default=0.15) 

	return parser.parse_args()