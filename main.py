from data_loader import MerckChallengeDataset
from net_structure import DNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

if __name__ == "__main__":

    EPOCH_NUM = 10
    BATCH_SIZE = 64

    data = MerckChallengeDataset("ACT7_competition_training.csv")
    train_data = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    net = DNN(4505, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

    for epoch in range(EPOCH_NUM):
	    for i, (inputs, labels) in enumerate(train_data):		
		    inputs, labels = Variable(inputs), Variable(labels)
		    y_pred = net(inputs)
		    loss = criterion(y_pred, labels)
		    print(EPOCH_NUM, i, loss)

		    optimizer.zero_grad()
		    loss.backward()
		    optimizer.step()





