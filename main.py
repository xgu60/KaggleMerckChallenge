from data_loader import MerckChallengeDataset
from net_structure import DNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":

	GPU_IN_USE = True
	EPOCH_NUM = 20
	BATCH_SIZE = 64


	data = MerckChallengeDataset("ACT7_competition_training.csv")
	train_data = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	net = DNN(4505, 1)
	if (GPU_IN_USE): 
		net.cuda()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

    #plot	   
	start = 0
	xdata = []
	ydata = []
	plt.ion()

	for epoch in range(EPOCH_NUM):
		for i, (inputs, labels) in enumerate(train_data):
			if (GPU_IN_USE):
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)
			y_pred = net(inputs)
			loss = criterion(y_pred, labels)
			#print(EPOCH_NUM, i, loss)		
			
			#update plot			
			start += 1
			xdata.append(start)
			ydata.append(loss.data[0] + 1)

			plt.plot(xdata, ydata, 'r-')
			plt.ylim(0, 1e4)
			plt.yscale('log')
			plt.pause(0.001)
			#plt.show() 

			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	plt.ioff()
	plt.show()


	 
		    

    

    





