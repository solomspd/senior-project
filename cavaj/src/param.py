import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Welcome to the Cavaj CLI decompilation tool')
	parser.add_argument("--data_point_num", help="the number of files to be used from the dataset", type=int, default=None)

	return parser.parse_args()