# -*- coding: utf-8 -*-
# Copyright 2017 The Wenchen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
data generator for long_mid_short term CNN

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import normalize
from network.model_config import LONG_TERM_LENGTH, TRAIN_TEST_SEP_RATE, MID_TERM_LENGTH, \
	EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
import torch


class DataGenerator_average(object):
	"""
	prepare (event) embedding and the label
	"""

	def __init__(self, dataframe, date_news_embedding, onehot_target=False):
		self.dataframe = dataframe
		self.date_news_embedding = np.array(date_news_embedding)

		self.date_news_embedding = self.date_news_embedding.reshape(self.date_news_embedding.shape[0],
		                                                            self.date_news_embedding.shape[1] *
		                                                            self.date_news_embedding.shape[2])

		self.one_hot_target = onehot_target
		# need to return those three
		self.date_list = []
		self.input_data = []
		self.label_data = []  # 2 dim for each label: long for(1,0),short for(0,1)

		self.get_data()

	def get_data(self):
		def one_hot_encode(int_label):
			assert int_label in (0, 1)
			return [[0, 1], [1, 0]][int_label]

		self.date_list_single = self.dataframe.values[:, 0][0:-1]  # for the last date we don't have label to train
		self.input_data_single = self.date_news_embedding[0:-1]
		self.input_data_single = normalize(self.input_data_single, axis=0)  # TODO remove sklearn normalize here
		self.label_data_single = self.dataframe.values[:, 2][1:]  # label is the next date up or down

		assert self.date_list_single.shape[0] == self.input_data_single.shape[0] == self.label_data_single.shape[0]
		# 30 __trading__ day perspective
		item_length = LONG_TERM_LENGTH

		for i, s in enumerate(self.date_list_single):  # index i is consistent
			if i == self.date_list_single.shape[0] - 1 - item_length: break
			self.date_list.append(
				list(reversed([self.date_list_single[i:i + item_length]])))  # reverse to match training index below
			self.input_data.append(list(reversed([self.input_data_single[i:i + item_length]])))
			if self.one_hot_target:
				self.label_data.append(one_hot_encode(self.label_data_single[i + item_length - 1]))  # notice the last
			else:
				self.label_data.append(self.label_data_single[i + item_length - 1])  # notice the last

		assert len(self.date_list) == len(self.input_data) == len(self.label_data), \
			(len(self.date_list), len(self.input_data), len(self.label_data))

	# return self.date_list, self.input_data, self.label_data

	def prepare_dataset(self):
		input_data_reshaped = np.array(
			[item[0] for item in self.input_data])  # TODO fix this step later , compress the 2nd dim that is 1

		separation_rate = TRAIN_TEST_SEP_RATE
		t = int(len(self.input_data) * separation_rate)
		input_train = input_data_reshaped[:t]
		input_test = input_data_reshaped[t:]
		label_train = self.label_data[:t]
		label_test = self.label_data[t:]

		# print ("input_train shape", np.array(input_train).shape)
		# print ("input_test shape", np.array(input_test).shape)
		# print ("label_train shape", np.array(label_train).shape)
		# print ("label_test shape", np.array(label_test).shape)

		# train data,  each item in the input_train has a full spectrum of 30 days news embedding
		short_term_train = np.array([item[0] for item in input_train])  # short term
		mid_term_train = np.array([item[:MID_TERM_LENGTH] for item in input_train])  # mid term
		long_term_train = np.array(input_train)  # long term
		label_train_array = np.array(label_train)
		# print ("short term shape", short_term_train.shape,
		# 	"mid term shape", mid_term_train.shape,
		# 	"long term shape", long_term_train.shape)

		train_data = {"short_term": short_term_train, "mid_term": mid_term_train, "long_term": long_term_train}
		# val data
		short_term_test = np.array([item[0] for item in input_test])  # short term
		mid_term_test = np.array([item[:MID_TERM_LENGTH] for item in input_test])  # mid term
		long_term_test = np.array(input_test)  # long term
		label_test_array = np.array(label_test)

		test_data = {"short_term": short_term_test, "mid_term": mid_term_test, "long_term": long_term_test}  # , label_test)

		return (train_data, label_train_array), (test_data, label_test_array)


class DataGenerator_average_torch(DataGenerator_average):
	def __init__(self, dataframe, date_news_embedding, onehot_target=False):
		super(DataGenerator_average_torch, self).__init__(dataframe=dataframe, date_news_embedding=date_news_embedding,
		                                                  onehot_target=onehot_target)

	def prepare_dataset_torch(self, cuda, batch_size):
		(train_data, train_targets), (test_data, test_targets) = self.prepare_dataset()
		x_train = train_data["long_term"]
		x_train = x_train.reshape((x_train.shape[0], 1, LONG_TERM_LENGTH, EMBEDDING_DIM * MAX_SEQUENCE_LENGTH))
		x_train = np.double(x_train)

		x_test = test_data["long_term"]
		x_test = x_test.reshape((x_test.shape[0], 1, LONG_TERM_LENGTH, EMBEDDING_DIM * MAX_SEQUENCE_LENGTH))
		x_test = np.double(x_test)

		train_features = torch.from_numpy(x_train).float()
		train_targets = torch.squeeze(torch.from_numpy(train_targets))

		test_features = torch.from_numpy(x_test).float()
		test_targets = torch.squeeze(torch.from_numpy(test_targets))

		print("train_features shape:", train_features.size(),
		      "train_targets shape", train_targets.size(),
		      "test_features shape", test_features.size(),
		      "test_targets shape", test_targets.size(),
		)

		kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

		train_set = torch.utils.data.TensorDataset(train_features, train_targets)
		test_set = torch.utils.data.TensorDataset(test_features, test_targets)

		train_loader = torch.utils.data.DataLoader(
			dataset=train_set,
			batch_size=batch_size, shuffle=True, **kwargs)

		test_loader = torch.utils.data.DataLoader(
			dataset=test_set,
			batch_size=batch_size, shuffle=True, **kwargs)

		return train_loader, test_loader
