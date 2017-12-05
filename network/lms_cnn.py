from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np

from network.model_config import *


class LMS_CNN(nn.Module):
	def __init__(self):
		super(LMS_CNN, self).__init__()
		self.mid_term_conv_layer = nn.Conv2d(1, 1, kernel_size=MID_TERM_CONV_KERNEL,padding=(3,0))  # MID_TERM_CONV_KERNEL)
		self.mid_conv_dropout = nn.Dropout()
		self.long_term_conv_layer = nn.Conv2d(1, 1, kernel_size=LONG_TERM_CONV_KERNEL)  # LONG_TERM_CONV_KERNEL)
		self.long_conv_dropout = nn.Dropout()
		self.fc1 = nn.Linear(3803, DENSE_HIDDEN_SIZE)  # TODO  fix this calculation later, cat -1 length
		self.fc1_dropout = nn.Dropout()
		self.fc2 = nn.Linear(DENSE_HIDDEN_SIZE, OUTPUT_DIM)

	def forward(self, x):
		short_term_input, mid_term_input, long_term_input = \
			x.narrow(2, 0, SHORT_TERM_LENGTH), x.narrow(2, 0, MID_TERM_LENGTH), x
		# https://github.com/pytorch/pytorch/issues/764
		short_term_input.contiguous()
		mid_term_input.contiguous()
		long_term_input.contiguous()

		short_term_flat = short_term_input.view(short_term_input.size(0), -1)

		mid_term_conv = self.mid_term_conv_layer(mid_term_input)
		mid_term_max_pool = F.max_pool2d(self.mid_conv_dropout(mid_term_conv), MID_TERM_POOL_SIZE)
		mid_term_flat = mid_term_max_pool.view(mid_term_max_pool.size(0), -1)

		long_term_conv = self.long_term_conv_layer(long_term_input)
		long_term_max_pool = F.max_pool2d(self.long_conv_dropout(long_term_conv), LONG_TERM_POOL_SIZE)
		long_term_flat = long_term_max_pool.view(long_term_max_pool.size(0), -1)

		# https://discuss.pytorch.org/t/different-dimensions-tensor-concatenation/5768
		mid_term_flat = mid_term_flat.view(mid_term_flat.size(0), -1)
		long_term_flat = long_term_flat.view(long_term_flat.size(0), -1)

		merge_layer = torch.cat([short_term_flat, mid_term_flat, long_term_flat], 1)

		# print (short_term_flat.size(), mid_term_max_pool.size(), long_term_max_pool.size())

		x = self.fc1(merge_layer)
		x = self.fc1_dropout(x)
		# F.dropout vs nn.dropout2d:https://discuss.pytorch.org/t/how-to-choose-between-torch-nn-functional-and-torch-nn-module-see-mnist-https-github-com-pytorch-examples-blob-master-mnist-main-py/2800/8
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)


class LMS_CNN_keras_wrapper:
	def __init__(self,cuda=True):
		self.cuda = cuda
		self._model = LMS_CNN()
		self.optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE, )
		if self.cuda:
			self._model.cuda()

	def train(self, train_loader):
		self._model.train()  # sets to train mode
		loss_list = []
		for batch_idx, (data, target) in enumerate(train_loader):
			if self.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			self.optimizer.zero_grad()
			output = self._model(data)
			loss = F.nll_loss(output, target)

			l2_reg = None
			for W in self._model.parameters():
				if l2_reg is None:
					l2_reg = W.norm(2)
				else:
					l2_reg = l2_reg + W.norm(2)
			loss += l2_reg * REGULARIZER_WEIGHT
			loss_list.append(loss.data)
			loss.backward()
			self.optimizer.step()
			# if batch_idx % args.log_interval == 0:
			# 	info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			# 		epoch, batch_idx * len(data), len(train_loader.dataset),
			# 		       100. * batch_idx / len(train_loader), loss.data[0])

		return np.mean(loss_list)

	def test(self, test_loader):
		self._model.eval()  # sets model to test mode
		test_loss = 0
		correct = 0
		for data, target in test_loader:
			if self.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target)
			output = self._model(data)
			test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
			pred = output.data.max(1, )[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		test_loss /= len(test_loader.dataset)
		info = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset))
		return info

	def fit(self, train_loader, test_loader, epochs):
		pbar = tqdm(range(1, epochs + 1))
		for epoch in pbar:
			train_info = "Training set: Average loss:"
			average_loss = self.train(train_loader)
			train_info += str(average_loss)

			test_info = self.test(test_loader)

			pbar.set_description(train_info + "|" + test_info)

	def save(self, path):
		torch.save(self._model.state_dict(), path)

	def load_weights(self, path):
		self._model.load_state_dict(torch.load(path))
