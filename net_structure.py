import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
	''' a simple dnn
	arguments: n_features e.g 1000
	           n_output e.g 1
	'''

	def __init__(self, n_features, n_output):
		super(DNN, self).__init__()
		self.h1 = nn.Linear(n_features, 2000)
		self.h1_dropout = nn.Dropout(p=0.25)
		self.h2 = nn.Linear(2000, 2000)
		self.h2_dropout = nn.Dropout(p=0.25)
		#self.h3 = nn.Linear(1000, 1000)
		#self.h3_dropout = nn.Dropout(p=0.25)
		self.output = nn.Linear(2000, n_output)

	def forward(self, x):
		x = F.relu(self.h1(x))
		x = self.h1_dropout(x)
		x = F.relu(self.h2(x))
		x = self.h2_dropout(x)
		#x = F.relu(self.h3(x))
		#x = self.h3_dropout(x)
		return self.output(x)


if __name__ == "__main__":
	net = DNN(777, 10)
	print(net)

