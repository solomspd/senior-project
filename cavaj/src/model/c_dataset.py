from torch.utils.data import Dataset

class dataset(Dataset):

	def __init__(self, ast, llc):
		super().__init__()
		self.ast = ast
		self.llc = llc
	
	def __len__(self):
		if len(self.ast) != len(self.llc):
			raise Exception("Length of AST and LLC are not equal")
		return len(self.ast)
	
	def __getitem__(self, index):
		return super().__getitem__(index)
		return self.ast[index], self.llc[index]