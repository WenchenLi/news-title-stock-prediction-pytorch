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
extract SVO based on textacy and spacy
for spacy model en_core_web_lg works the best.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textacy
import spacy

from network.util import save_pickle


def get_svo(filepath):
	"""
	extract svo from sentences in the text file,
	where each line has one sentence. The extracted result
	is saved to svo_result.pickle. For each extracted
	sentence, the sentence index from text file is saved
	along with its extracted result.
	For example:
	sent:egypt salafists plan rally if presidential candidate barred.
	result: (6, ['egypt salafists', 'plan', 'rally'])

	:param filePath: str| file path to sentences to be extracted

	"""
	# load txt file
	all_lower_title = []
	with open(filepath, "r") as fin:
		for k, i in enumerate(fin):
			all_lower_title.append(unicode(i.lower(),'utf-8'))

	# load spacy model, notice we use the large model for possibly better performance
	model='en_core_web_lg'
	nlp = spacy.load(model)

	# start process
	svo_results = []
	len_titles = len(all_lower_title)

	for n,title in enumerate(all_lower_title):
		all_lower_title_unicode = title
		docs = nlp(all_lower_title_unicode)
		b = textacy.extract.subject_verb_object_triples(docs)
		svo = list(b)
		if len(svo) > 0:
			i = svo[0]
			svo_results.append((n,[str(j) for j in i])) # item in i still associates with spacy, str to remove it

	print ("total sents:",len_titles,"extracted sents:",len(svo_results),
		"extraction_rate:", len(svo_results)/float(len_titles))
	save_pickle(svo_results,"training_dir","svo_result")


file_path = 'training_dir/titles.txt'
get_svo(file_path) 	# 421738 188347 extraction_rate: 0.446597176446
