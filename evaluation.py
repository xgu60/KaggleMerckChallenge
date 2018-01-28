import torch
import numpy as np



def pearson_r_square(data_pred, data_label):
	label_mean = torch.mean(data_label)
	pred_mean = torch.mean(data_pred)
	numerator = torch.sum((data_pred - pred_mean) * (data_label - label_mean)) ** 2
	denominator = torch.sum(torch.pow(data_pred - pred_mean, 2)) * torch.sum(torch.pow(data_label - label_mean, 2))
	return numerator / denominator

def pearson_r(data_pred, data_label):
	label_mean = torch.mean(data_label)
	pred_mean = torch.mean(data_pred)
	numerator = torch.sum((data_pred - pred_mean) * (data_label - label_mean))
	denominator = np.sqrt(torch.sum(torch.pow(data_pred - pred_mean, 2)) * 
		torch.sum(torch.pow(data_label - label_mean, 2)))
	return numerator / denominator

if __name__ == "__main__":
	
	data_pred = torch.Tensor([2.0, 3, 4])
	data_label = torch.Tensor([3, 5, 4])
	print (data_pred)
	print (data_label)
	
	print(pearson_r_square(data_pred, data_label))
	print(pearson_r(data_pred, data_label))

