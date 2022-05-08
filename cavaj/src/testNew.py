import inspect
import re
import sys
import logging
from pathlib import Path
import matplotlib
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import networkx as nx 
import javalang
import random
from tqdm import tqdm

with open ("/home/g07/andrew/senior-project/cavaj/src/test.txt", 'r') as file:
    l = file.readlines()
    new_list = [x for x in l if (x.find('"') and not x[0].isdigit())]
with open('newTokens.txt', 'w') as testFile:
			for item in new_list:
				testFile.write("%s" % item)    
print(len(new_list))
