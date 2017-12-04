# -*- coding: utf-8 -*-
# Copyright 2017 The Wenchen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Preprocess data to send into models: Neural Tensor Network, Deep Prediction Network

news dataset for training own corpus word embedding
dataset can be access via: https://drive.google.com/open?id=0B3C8GEFwm08QY3AySmE2Z1daaUE


Neural Tensor Network:
1. get all the titles and build a vocabulary
2. prepare the embedding layer for the model as embedding lookup layer

Deep Prediction Network:
prepare word embedding (layer) to the training|testing model
ways to handle unknown words:https://groups.google.com/forum/#!topic/word2vec-toolkit/TgMeiJJGDc0
Now __UNK__ vector is random, __PAD__ use zeros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import re
import pandas as pd
from collections import defaultdict

from network.util import load_pickle, save_pickle, add_one_day, hour_12_to_24
from network.model_config import TRAINING_DIR


def get_all_news_raw_text(news_path="data/news"):
	"""
	get raw text from all texts from the dataset under the same folder, save all the text named
	news.txt under folder data.
	note this method only applies to data for Reuters and bloomberg datasets
	mentioned in the paper:
	Ding, Xiao, Yue Zhang, Ting Liu and Junwen Duan. “Deep Learning for Event-Driven Stock Prediction.” IJCAI (2015).
	:param news_path: str| path to the extracted news data folder
	"""
	train_embedding_input_data = []  # one news as one item in list
	for root, subFolders, files in os.walk(news_path):
		if len(root.split("/")) != 4: continue  # only work on root end with subfolder with date
		for file in files:
			filepath = root + "/" + file
			temp_list = []
			with open(filepath) as fin:
				for line in fin:
					temp_list.append(line)
			train_embedding_input_data.append("".join(temp_list))
	# write all texts to 1 line txt file
	with open(news_path + ".txt", "w") as ftxtout:
		ftxtout.write("".join(train_embedding_input_data))


def get_data_dict(news_path="data/news"):
	"""
		get all news title into a dictionary(list), with date as key, values at list of item,where
		item[0] is news title, item[1] is news reported time[hour-min-sec]. A example would be
		dict[20170101] = [["news_title_1","00:00:00"],["news_title_2","00:00:00"]]
		save data_dict as data.pickle for later use
		:param news_path:news_path: str| path to the extracted news data folder
	"""
	data = defaultdict(list)
	for root, subFolders, files in os.walk(news_path):
		if len(root.split("/")) != 4: continue  # only work on root end with subfolder with date
		date = root.split("/")[-1]
		if "bloomberg" in root:  # bloomberg format
			# print "bloomberg"
			date = date.replace("-", "")
			for file in files:
				filepath = root + "/" + file
				with open(filepath) as fin:
					news_tuple = []
					for i, line in enumerate(fin):
						try:  # original file format not consistent
							if i == 0:  # title
								title = line[3:-1]
								news_tuple.append(title)
							elif i == 2:  # time
								time = line[14:-2]
								hour = int(time[:2].replace(" ", "")) + 9
								date_add_one = hour > 24
								if date_add_one:
									hour %= 24
									date_one_added = add_one_day(date)
									time = str(hour) + time[2:]
								news_tuple.append(time)
								data[date_one_added if date_add_one else date].append(news_tuple)
								break  # stop iterate current file
						except:
							if i == 1:  # title
								title = line[:-1]
								news_tuple.append(title)
							elif i == 5:  # time
								time = line[11:-2]
								hour = int(time[:2]) + 9
								date_add_one = hour > 24
								if date_add_one:
									hour %= 24
									date_one_added = add_one_day(date)
									time = str(hour) + time[2:]
								news_tuple.append(time)
								data[date_one_added if date_add_one else date].append(news_tuple)
								break  # stop iterate current file

		if "ReutersNews" in root:  # reuter format
			# print "reuter"
			for file in files:
				filepath = root + "/" + file
				with open(filepath) as fin:
					news_tuple = []
					for i, line in enumerate(fin):
						if i == 0:  # title
							title = line[3:-1]
							# print title
							news_tuple.append(title)
						elif i == 2:  # time
							time = line.split(" ")[-2]
							time = hour_12_to_24(time)
							news_tuple.append(time)
							data[date].append(news_tuple)
							break  # stop iterate current file
	# within a day sort result
	for k in data:
		data[k].sort(key=lambda tup: tup[1])
	save_pickle(data, "training_dir", "data")


def clean_str(string):
	"""
		Tokenization/string cleaning for all datasets except for SST.
		Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()


def get_texts_list(titles_file_path):
	"""
		get titles in the format of list of strings, with each title an item
		in the list
		:param titles_file_path:str| path to title file which store title as text file , each line for one
		:return:texts|list of strings, raw text to be processed
	"""
	texts = []
	with open(titles_file_path, "r") as tf:
		for line in tf:
			texts.append(line.strip())
	return texts


def embedding2dict(embedding_full_path):
	"""
		load txt embedding into dict with key the word, value the vector in np.array
		:param embedding_full_path: str| path to embedding txt file
		:return: embedding_index| dict,  key the word, value the vector in np.array
	"""
	print('Indexing word vectors.')
	embeddings_index = {}
	f = open(embedding_full_path, 'r')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Found %s word vectors.' % len(embeddings_index))
	return embeddings_index


def load_structured_events(structured_events_pickle):
	"""
		load structured events into texts_O1,  texts_Pred, texts_O2 , datetimes, where the former 3 will be
		converted by tokenizer into ids.

		:param structured_events_pickle:
		:return:texts_O1:list of str | Object data
						texts_Pred: list of str | Predicate data
						texts_O2: list of str | Subject data
						datetimes:list of str | datetimes of events
	"""

	texts_O1 = []
	texts_Pred = []
	texts_O2 = []
	datetimes = []

	# /Users/wenchenli/PycharmProjects/event-driven-stock-prediction/training_dir/svo_datetime.pickle
	svo_datetime = load_pickle(structured_events_pickle)

	for svod in svo_datetime:
		texts_O1.append(" ".join(svod[0]))
		texts_Pred.append(" ".join(svod[1]))
		texts_O2.append(" ".join(svod[2]))
		datetimes.append(svod[3])

	return texts_O1, texts_Pred, texts_O2, datetimes


def get_long_short_signal(input_dataframe):
	"""
	get long or short signal from the csv file 1st col date, 2nd col adj_close_price,
	3rd col list of timed news titles
	to be more specific 3rd col looks like:
	[['Hey buddy, can you spare $600 for a Google share?', '16:25:00'],
	['Exxon Mobil offers plan to end Alaska dispute', '18:15:00']]
	:param input_dataframe: input dataframe with 1
	:return:pd.dataframe| 1st col date, 2nd col adj_close_price, 3rd col long_short_signal,
								4th col list of timed news titles.
	"""

	adj_close = input_dataframe.values[:, 1]
	signals = []
	for i, c in enumerate(adj_close):
		if i == 0:
			signals.append(0)  # append what ever to make dataformat consistent
			continue
		else:
			signal = 0 if c < adj_close[i - 1] else 1
			signals.append(signal)

	input_dataframe['long short signal'] = pd.Series(np.array(signals), index=input_dataframe.index)
	input_dataframe = input_dataframe[["Date", "Adj Close","long short signal","news_title"]]
	return input_dataframe


def get_all_news_titles_to_txt(input_dataframe):
	"""
	get all news titles into text file saved as titles.txt under TRAINING_DIR
	:param input_dataframe: dataframe| 1st col date, 2nd col adj_close_price,
	3rd col list of timed news titles
	"""
	timed_news_titles = input_dataframe["news_title"].tolist()

	with open(TRAINING_DIR+"/"+"all_titles.txt","w") as fo:
		for daily_timed_news_titles in timed_news_titles:
			for daily_timed_news_title in daily_timed_news_titles:
				fo.write(daily_timed_news_title[0] + "\n")#index 0 points to titles, index 1 points to time


def add_news_title_to_dataframe(df_dt_adjclose, data_pickle_path):
	"""
	:param df_dt_adjclose: pd.dataframe| 1st col date, 2nd col adj_price, 3rd col long_short_signal
	:param data_pickle_path: str| loaded data is defaultdict(list) with key in date, value as lists of
													news each list contains 2 element [news_title, news_time]
	:return: df_dt_adjclose:  pd.dataframe | add news corresponding to each date, intra-day news should be
																sorted by time also
	"""

	def date_format_fit_dataframe(dt_str_data_pickle):
		return dt_str_data_pickle[:4] + "-" + dt_str_data_pickle[4:6] + "-" + dt_str_data_pickle[6:]

	def date_format_fit_data_pickle(dt_str_dataframe):
		return dt_str_dataframe.replace("-", "")

	data = load_pickle(data_pickle_path)

	date = df_dt_adjclose.values[:, 0]

	news_title = []

	for d in date:
		news_sorted_list = data[date_format_fit_data_pickle(d)]
		news_title.append(news_sorted_list)

	df_dt_adjclose['news_title'] = pd.Series(news_title, index=df_dt_adjclose.index)

	return df_dt_adjclose


def prepare_embedding_on_date(df_dt_adjclose_with_titles, embedding_matrix, vocab_processor, option="mean"):
	"""
		prepare news embedding for each date, option mean use average intraday titles embedding, which is average along
		titles to get an "average" title embedding. option all, keeps all the title.

		:param option: str| options for embedding, mean for average each date news embedding,
																								all for keep each date news title separately
		:type vocab_processor: VocabularyProcessor| vocab processor helps to transform texts to ids
		:param df_dt_adjclose_with_titles:dataframe |1st col date, 2nd col adj_price, 3rd col long_short_signal,
																								4th col list of sorted intra-day news.
		:param embedding_matrix:np.array| embedding matrix of which word index is consistent with vocab_processor
		:return:list| date news embedding
	"""
	assert option in ["mean", "all"]

	date_news_embedding = []
	news_titles = df_dt_adjclose_with_titles.values[:, 3]

	for news_title in news_titles:
		x_text_cleaned = [clean_str(x[0]) for x in news_title]  # 0 refer to text, 1 refer to time
		x_text_cleaned_transformed = vocab_processor.transform(x_text_cleaned)  # check pad

		sentence_embeddings = [[embedding_matrix[i] for i in ids] for ids in x_text_cleaned_transformed]  # fix detail later

		if option == "mean":
			date_embedding = np.mean(sentence_embeddings, axis=0)
		else:  # not averaging titles
			date_embedding = sentence_embeddings

		date_news_embedding.append(date_embedding)

	return date_news_embedding
