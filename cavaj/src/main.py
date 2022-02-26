import torch
import dgl
from pathlib import Path

from data_proc import data_proc

if __name__ == '__main__':
	dataset_path = Path("../data/50k")
	data_processor = data_proc()
	trg,srcf,srcg,srcllc = data_processor.load_data(dataset_path / "java_src", dataset_path / "bytecode")
	