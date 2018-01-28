import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class MerckChallengeDataset(Dataset):
	'''read data from CSV file
	   Argument: CSV file name
	'''
	def __init__(self, csv_path):
		xy = pd.read_csv(csv_path)		
		self.len = int(xy.shape[0] * 0.7)		
		self.x_data = torch.from_numpy(xy.ix[: self.len - 1, 2:].as_matrix()).float()
		self.y_data = torch.from_numpy(xy.ix[: self.len - 1, 1].as_matrix()).float().view(self.len, 1)

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

def getTestData(csv_path):
	xy = pd.read_csv(csv_path)
	st = int(xy.shape[0] * 0.7)
	rows = xy.shape[0] - st
	x_data = torch.from_numpy(xy.ix[st:, 2:].as_matrix()).float()
	y_data = torch.from_numpy(xy.ix[st:, 1].as_matrix()).float().view(rows, 1)
	return x_data, y_data


if __name__ == "__main__":
	data = MerckChallengeDataset("ACT7_competition_training.csv")
	print(data.__len__())	
	
	x, y = getTestData("ACT7_competition_training.csv")
	print(x.size())
	print(y.size())
	
