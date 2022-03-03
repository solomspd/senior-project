import torch
from torch.utils.data import DataLoader, Dataset
import dgl
import networkx as nx
from pathlib import Path

from logging import exception
from collections import defaultdict
from subprocess import list2cmdline

from data_proc import data_proc
import param

from model.c_dataset import dataset


def text_data_collator(dataset: Dataset):
	def collate(data):
		batch = defaultdict(list)
		for datum in data:
			for name, field in dataset.fields.items():
				batch[name].append(field.preprocess(getattr(datum, name)))
		batch = {name: field.process(batch[name])
				for name, field in dataset.fields.items()}
		return batch
	return collate


if __name__ == '__main__':
	arg = param.parse_args()
	dataset_path = Path("../data/50k")
	trg_ast,trg_llc,_,_ = data_proc(arg).load_data(dataset_path / "java_src", dataset_path / "bytecode")

	#logging = get_logger(log_path=os.path.join(args.log_dir, "log" + time.strftime('%Y%m%d-%H%M%S') + '.txt'), print_=True, log_=True)

	# Solom & botta todo
	#SEED = 1234 
    #torch.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = True
    
	data_set = dataset(trg_llc,trg_ast)

	trn_prop = int(len(data_set)*0.8)
	val_prop = int((len(data_set)) - trn_prop * 0.25)
	tst_prop = int(len(data_set) - trn_prop - val_prop)
	train, validate, test = torch.utils.data.random_split(data_set, [trn_prop, val_prop, tst_prop])

	#logging("Baby cavaj is training now!")
    #logging("Number of training examples: %d" % (len(train.examples)))
    #logging("Number of validation examples: %d" % (len(validate.examples)))
    #logging("Number of testing examples: %d" % (len(test.examples)))
    #logging("Unique tokens in source assembly vocabulary: %d " % (len(SRC.vocab)))
    #logging("Max input length : %d" % (max_len_src))
    #logging("Max output length : %d" % (max_len_trg))

	# model = GCN(arg, trg_ast, trg_llc)

	workers = 0
	batchSize = 1
	collate = text_data_collator(train)
	train_iter = DataLoader(train, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	collate = text_data_collator(validate)
	valid_iter = DataLoader(validate, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	collate = text_data_collator(test)
	test_iter = DataLoader( test, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
