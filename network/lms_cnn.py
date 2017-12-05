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

from network.model_config import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
										help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
										help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
										help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, metavar='LR',
										help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
										help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
										help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
										help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
										help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
		torch.cuda.manual_seed(args.seed)


class LMS_CNN(nn.Module):
	def __init__(self):
		super(LMS_CNN, self).__init__()
		self.mid_term_conv_layer = nn.Conv2d(1, 1, kernel_size=MID_TERM_CONV_KERNEL)  # MID_TERM_CONV_KERNEL)
		self.long_term_conv_layer = nn.Conv2d(1, 1, kernel_size=LONG_TERM_CONV_KERNEL)  # LONG_TERM_CONV_KERNEL)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(3608, 16)  # cat -1 length
		self.fc2 = nn.Linear(16, 2)

	def forward(self, x):
		short_term_input, mid_term_input, long_term_input = \
				x.narrow(2, 0, SHORT_TERM_LENGTH), x.narrow(2, 0, MID_TERM_LENGTH), x
		# https://github.com/pytorch/pytorch/issues/764
		short_term_input.contiguous()
		mid_term_input.contiguous()
		long_term_input.contiguous()

		short_term_flat = short_term_input.view(short_term_input.size(0), -1)

		# mid_term_input = mid_term_input.view(mid_term_input.size(0),mid_term_input.size(2),mid_term_input.size(3))
		mid_term_conv = self.mid_term_conv_layer(mid_term_input)
		mid_term_max_pool = F.max_pool2d(mid_term_conv, MID_TERM_POOL_SIZE)
		mid_term_flat = mid_term_max_pool.view(mid_term_max_pool.size(0), -1)

		# long_term_input = long_term_input.view(long_term_input.size(0),long_term_input.size(2),long_term_input.size(3))
		long_term_conv_out = F.max_pool2d(self.long_term_conv_layer(long_term_input), LONG_TERM_POOL_SIZE)
		long_term_flat = long_term_conv_out.view(long_term_conv_out.size(0), -1)

		# https://discuss.pytorch.org/t/different-dimensions-tensor-concatenation/5768
		mid_term_flat = mid_term_flat.view(mid_term_flat.size(0), -1)
		long_term_flat = long_term_flat.view(long_term_flat.size(0), -1)

		# print("short:", short_term_flat.size(),
		#       "mid:", mid_term_flat.size(),
		#       "long:", long_term_flat.size())

		merge_layer = torch.cat([short_term_flat, mid_term_flat, long_term_flat], 1)

		x = self.fc1(merge_layer)
		x = self.fc2(x)
		x = F.dropout(x, training=self.training) # F.dropout vs nn.dropout2d:https://discuss.pytorch.org/t/how-to-choose-between-torch-nn-functional-and-torch-nn-module-see-mnist-https-github-com-pytorch-examples-blob-master-mnist-main-py/2800/8
		return F.log_softmax(x,dim=1) #TODO check dim =0 correctnetss


class LMS_CNN_keras_wrapper:
	def __init__(self):
		self._model = LMS_CNN()
		self.optimizer = optim.Adam(self._model.parameters(), lr=args.lr, )
		if args.cuda:
				self._model.cuda()

	def train(self, epoch, train_loader):
		self._model.train()  # sets to train mode
		for batch_idx, (data, target) in enumerate(train_loader):
			if args.cuda:
					data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			self.optimizer.zero_grad()
			output = self._model(data)
			# print ("output size:",output.size(),"target size:",target.size())
			loss = F.nll_loss(output, target)

			l2_reg = None
			for W in self._model.parameters():
				if l2_reg is None:
					l2_reg = W.norm(2)
				else:
					l2_reg = l2_reg + W.norm(2)
			loss += l2_reg * REGULARIZER_WEIGHT

			loss.backward()
			self.optimizer.step()
			if batch_idx % args.log_interval == 0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
							epoch, batch_idx * len(data), len(train_loader.dataset),
										 100. * batch_idx / len(train_loader), loss.data[0]))

	def test(self, test_loader):
		self._model.eval()  # sets model to test mode
		test_loss = 0
		correct = 0
		for data, target in test_loader:
			if args.cuda:
					data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target)
			output = self._model(data)
			test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
			pred = output.data.max(1, )[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		test_loss /= len(test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))

	def fit(self, train_loader, test_loader, epochs):  # batch_size=BATCH_SIZE):
		# TODO keras prepare batch within model, torch prepare batch in datagenerator due to former static
		# TODO and latter dynamic in the computation graph
		for epoch in range(1, epochs + 1):
				self.train(epoch, train_loader)
				self.test(test_loader)

	def save(self,path):
		torch.save(self._model.state_dict(),path)

	def load_weights(self,path):
		self._model.load_state_dict(torch.load(path))