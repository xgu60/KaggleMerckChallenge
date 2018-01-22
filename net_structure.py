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
		self.h1 = nn.Linear(n_features, 1000)
		self.h2 = nn.Linear(1000, 1000)
		self.h3 = nn.Linear(1000, 1000)
		self.output = nn.Linear(1000, n_output)

	def forward(self, x):
		x = F.relu(self.h1(x))
		x = F.relu(self.h2(x))
		x = F.relu(self.h3(x))
		return self.output(x)


if __name__ == "__main__":
	net = DNN(777, 10)
	print(net)

