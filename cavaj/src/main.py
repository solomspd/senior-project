from collections import defaultdict
from logging import exception
from pathlib import Path
from subprocess import list2cmdline

import networkx as nx
import torch
from torch.utils.data import DataLoader, Dataset

import param
from data_proc import data_proc
from model.c_dataset import dataset
from model.model_top import cavaj
from model.utils import NoamOpt


if __name__ == '__main__':
	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	trg_ast,trg_llc = data_proc(arg).load_data(dataset_path / "java_src", dataset_path / "bytecode")

	#logging = get_logger(log_path=os.path.join(args.log_dir, "log" + time.strftime('%Y%m%d-%H%M%S') + '.txt'), print_=True, log_=True)

	#SEED = 1234 
    #torch.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = True
    
	data_set = dataset(trg_llc,trg_ast)

	# this is ugly please change it
	trn_prop = int(len(data_set)*0.8)
	val_prop = int((len(data_set)) - trn_prop * 0.25)
	tst_prop = int(len(data_set) - trn_prop - val_prop)

	# vocab_len = torch.count_nonzero(torch.bincount(torch.cat([i.x.flatten() for i in trg_llc])))
	vocab_len = torch.max(torch.cat([i.x.flatten() for i in trg_llc]))

	train, validate, test = torch.utils.data.random_split(data_set, [trn_prop, val_prop, tst_prop])

	model = cavaj(arg, trg_ast, trg_llc)

	workers = 0
	batchSize = 1
	train_iter = DataLoader(train, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	valid_iter = DataLoader(validate, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	test_iter = DataLoader( test, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)

	# moved training to main
	# optim = NoamOpt(arg.hid_dim, arg.lr_ratio, arg.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) # use NoamOpt from attention is all you need
	crit = torch.nn.CrossEntropyLoss()

	model(trg_ast[0], trg_llc[0])

	for i in arg.epochs:
		pass

	model.train(train_iter, valid_iter)