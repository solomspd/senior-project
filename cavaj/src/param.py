import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Welcome to the Cavaj CLI decompilation tool')
	parser.add_argument("--data_point_num", help="the number of files to be used from the dataset", type=int, default=None)
	parser.add_argument("--batch_sz", help="the batch size to be used for training", type=int, default=1)
	parser.add_argument("--epochs", help="the number of epochs to be used for training", type=int, default=1000)
	parser.add_argument("--warmup", help="NoamOpt optimizer warmup", type=int, default=8000)
	parser.add_argument('--hid_dim', type=int, default=256)
	parser.add_argument('--mem_dim', type=int, default=64)
	parser.add_argument('--encdec_units', type=int, default=2) 
	parser.add_argument('--n_embed', type=int, default=2) 
	parser.add_argument('--n_heads', type=int, default=4)
	parser.add_argument('--pf_dim', type=int, default=512)
	parser.add_argument('--max_tolerate_len', type=int, default=1200) #for small GPU mem 
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--depth_dim', type=int, default=40)
	parser.add_argument('--output_dim', type=int, default=128)
	parser.add_argument('--graph_aug', action='store_true', default=True)
	parser.add_argument('--embedding_flag',type=int, default=1)
	parser.add_argument('--lr_rate', type=float, default=0.01) 
	parser.add_argument('--no_prog', help='disable progress bars', action='store_true', default=False) 
	parser.add_argument('--ast_max_len', help='The maximum possible length of the resulting AST', type=int, default=100)
	parser.add_argument('--llc_max_len', help='The maximum possible length of the input LLC', type=int, default=500)
	parser.add_argument('--profile', help='Profile model to tensorboard', action='store_true', default=False)
	parser.add_argument('--checkpoint', help='File to pick up training from again', type=str, default='/home/g07/senior-project/cavaj/model_checkpoints/2022-05-10T15:22:35')

	return parser.parse_args()