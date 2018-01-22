import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class MerckChallengeDataset(Dataset):
	'''read data from CSV file
	   Argument: CSV file name
	'''
	def __init__(self, csv_path):
		xy = pd.read_csv(csv_path)
		self.len = xy.shape[0]
		self.x_data = torch.from_numpy(xy.ix[:, 2:].as_matrix()).float()
		self.y_data = torch.from_numpy(xy.ix[:, 1].as_matrix()).float().view(xy.shape[0], 1)

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


if __name__ == "__main__":
	xy = pd.read_csv("ACT7_competition_training.csv")
	y_data = torch.from_numpy(xy.ix[:, 1].as_matrix()).float().view(xy.shape[0], 1)

	print(y_data)
	print(y_data.shape)
	#data = MerckChallengeDataset("ACT7_competition_training.csv")
	#print(data.__len__())
	#print(data.__getitem__(1))

