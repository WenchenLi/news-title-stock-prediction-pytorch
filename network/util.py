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
util for the project:

word embedding: neural tensor network input look up
Learn the initial work representation of the d-dimensions d=100, from large-scale financial news corpus,
using skip gram.
(TODO , worry about this later on the source, since we have good word vector now)

Triplet extraction from text: input : given segmented sentence, output: triplet(single word) + (time)
1. openIE : use ReVerb [Fader et al., 2011] to extract the candidate tuples of the event (O1' , P ' , O2'),
and
2. SRL: then parse the sentence with ZPar [Zhang and Clark, 2011] to extract the subject, object and predicate.
assume that O1′ , O2′ , and P ′ should contain the subject, object, and predicate, respectively.
If this is not the case, the candidate tuple is filtered out.
Redundancy in large news data allows this method to capture major events with high recalls.
some suggestion: https://stackoverflow.com/questions/8063334/extract-triplet-subject-predicate-and-object-sentence
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import datetime


def add_one_day(date_string):
	"""
	given date string like 20100314 logiclly add one day and return the
	same format 20100315
	:param date_string:
	:return: date_string
	"""
	year = int(date_string[:4])
	month = int(date_string[4:6])
	day = int(date_string[6:])
	date = datetime.datetime(year, month, day)
	date += datetime.timedelta(days=1)
	return date.strftime("%Y%m%d")


def hour_12_to_24(time_12):
	"""
	given input like 8:17pm, output time in 24 hour with additional seconds 20:17:00
	:param time_12:input like 8:17pm
	:return:output time in 24 hour with additional seconds 20:17:00
	"""
	hour = time_12.split(":")[0]
	min = time_12.split(":")[1][:2]

	if "am" == time_12[-2:]:
			hour = "0"+hour if int(hour) < 10 else hour
	elif "pm" ==time_12[-2:]:
			hour =  str(int(hour)+12)
	time_24 = hour+":"+min+":"+"00"
	return time_24


def data_dict_to_txt(data_dict, txt_path):
	"""
	convert data_dict into txt format that will fit openIE 5.0 extraction
	txt format: 1st col title, 2nd col date$time, openIE use 1st col
	:param data_dict:the data preprocessed by get_data_dict
	:param txt_path: the text file saved path
	"""
	data = load_pickle(data_dict)
	titles = open(txt_path+"/"+"titles.txt","w")
	datetimes = open(txt_path+"/"+"datetimes.txt","w")
	date_time_seperator = "$"
	titles_buff = []
	datetimes_buff = []
	for date in data:
		for item in data[date]:
			title = item[0]
			time = item[1]
			titles_buff.append(title+".\n")
			datetimes_buff.append(date + date_time_seperator + time + "\n")

	titles.writelines(titles_buff)
	datetimes.writelines(datetimes_buff)

	titles.close()
	datetimes.close()


def save_pickle(a,folder,name):
	"""
	save object to pickle
	"""
	with open(folder +"/"+ name + '.pickle', 'wb') as handle:
		pickle.dump(a, handle)


def load_pickle(filepath):
	"""
	load object from pickle
	"""
	with open(filepath, 'rb') as handle:
		return pickle.load(handle)
