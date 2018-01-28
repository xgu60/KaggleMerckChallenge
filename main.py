from data_loader import MerckChallengeDataset, getTestData
from net_structure import DNN, DNN_simple
from evaluation import pearson_r_square, pearson_r
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":

	VISUAL_ON = True
	GPU_IN_USE = True
	EPOCH_NUM = 20
	BATCH_SIZE = 256


	data = MerckChallengeDataset("ACT7_competition_training.csv")	
	train_data = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	test_x_data, test_y_data = getTestData("ACT7_competition_training.csv")

	net = DNN(4505, 1)
	if (GPU_IN_USE): 
		net.cuda()
		test_data = Variable(test_x_data).cuda()
		test_label = Variable(test_y_data).cuda()
	else:
		test_data = Variable(test_x_data)
		test_label = Variable(test_y_data)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

	if VISUAL_ON:
		start = 0
		xdata = []
		train = []
		test = []
		plt.ion()

	for epoch in range(EPOCH_NUM):
		for i, (inputs, labels) in enumerate(train_data):
			net.train()
			if GPU_IN_USE:
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)
			y_pred = net(inputs)
			loss = criterion(y_pred, labels)					

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			#evaluate the trained network
			net.eval()
			y_pred = net(inputs)
			train_error = pearson_r_square(y_pred.data, labels.data)

			test_pred = net(test_data)
			test_error = pearson_r_square(test_pred.data, test_label.data)
			#print(EPOCH_NUM, i, r2)

			if VISUAL_ON:
				#update plot			
				start += 1
				xdata.append(start)
				train.append(train_error)
				test.append(test_error)

				plt.plot(xdata, train, 'b-')
				plt.plot(xdata, test, 'r-')
				#plt.ylim(0, 1e4)
				#plt.yscale('log')
				plt.pause(0.001)

	if VISUAL_ON:
		plt.ioff()
		plt.show()


	 
		    

    

    





