from subprocess import list2cmdline
import torch
import dgl
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from data_proc import data_proc
import param
from torch.utils.data import DataLoader, Dataset
from logging import exception
from collections import defaultdict

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
	trg_ast,src_f,src_g,trg_llc = data_proc(arg).load_data(dataset_path / "java_src", dataset_path / "bytecode")
	plt.figure()
	nx.draw(dgl.to_networkx(trg_ast[0]))
	plt.show()
	#logging = get_logger(log_path=os.path.join(args.log_dir, "log" + time.strftime('%Y%m%d-%H%M%S') + '.txt'), print_=True, log_=True)

	# Solom & botta todo
	#SEED = 1234 
    #torch.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = True
    
	dataSet = []
	source = []
	target =[]
	llcGraph = []

	data_sets = Dataset(list(zip(trg_llc,trg_ast)))
	train, validate, test = data_sets.split([0.8, 0.05, 0.15])

	#logging("Baby cavaj is training now!")
    #logging("Number of training examples: %d" % (len(train.examples)))
    #logging("Number of validation examples: %d" % (len(validate.examples)))
    #logging("Number of testing examples: %d" % (len(test.examples)))
    #logging("Unique tokens in source assembly vocabulary: %d " % (len(SRC.vocab)))
    #logging("Max input length : %d" % (max_len_src))
    #logging("Max output length : %d" % (max_len_trg))

	workers = 0
	batchSize = 1
	collate = text_data_collator(train)
	train_iter = DataLoader(train, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	collate = text_data_collator(validate)
	valid_iter = DataLoader(validate, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)
	collate = text_data_collator(test)
	test_iter = DataLoader( test, batch_size=batchSize, collate_fn=None, num_workers=workers, shuffle=False)