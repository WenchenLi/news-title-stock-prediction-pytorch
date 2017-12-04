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
pre-process data into format that can be sent into deep prediction model

saved input_dataframe_with_signal and date_news_embedding for training,
saved embedding_matrix, vocab_processor for online prediction
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network.model_config import EMBEDDING_METHOD, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MIN_FREQ_FILTER
from network.preprocess import *
import nlp_vocab

if not os.path.exists(TRAINING_DIR):
	os.makedirs(TRAINING_DIR)

# input
SPY_CSV_PATH = 'data/SPY.csv'
Input_dataframe_path = TRAINING_DIR + "/" + "input_dataframe.pickle"
embedding_vector_path = "data/embedding/" + EMBEDDING_METHOD + "/" + "model.vec"

# intermediate output
data_pickle_path = TRAINING_DIR + "/" + "data.pickle"
TITLES_PATH = TRAINING_DIR + "/" + "all_titles.txt"
Vocab_path = TRAINING_DIR + "/" + "vocab_processor"
embedding_matrix_filename = "embedding_matrix"
input_dataframe_with_signal_filename = "input_dataframe_with_signal"
date_news_embedding_filename = "date_news_embedding_" + EMBEDDING_METHOD + "_" + str(EMBEDDING_DIM)

# embedding Preparation for deep prediction model
# ==================================================

input_dataframe = load_pickle(Input_dataframe_path)
get_all_news_titles_to_txt(input_dataframe)
input_dataframe_with_signal = get_long_short_signal(input_dataframe)

# Load data
if os.path.isfile(Vocab_path):
	print("Loading Vocabulary ...")
	vocab_processor = nlp_vocab.VocabularyProcessor.restore(Vocab_path)

else:
	print("Building Vocabulary ...")
	x_text = get_texts_list(TITLES_PATH)
	x_text_cleaned = [clean_str(x) for x in x_text]

	# Build/load vocabulary
	max_document_length = MAX_SEQUENCE_LENGTH
	min_freq_filter = MIN_FREQ_FILTER

	vocab_processor = nlp_vocab.VocabularyProcessor(max_document_length, min_frequency=min_freq_filter)
	vocab_processor.fit(x_text_cleaned)
	vocab_processor.save(Vocab_path)
	print ("vocab_processor saved at:", Vocab_path)

embedding_matrix = vocab_processor.prepare_embedding_matrix(embedding_vector_path)
save_pickle(embedding_matrix, TRAINING_DIR, embedding_matrix_filename)

date_news_embedding = prepare_embedding_on_date(input_dataframe_with_signal, embedding_matrix, vocab_processor)

save_pickle(input_dataframe_with_signal, TRAINING_DIR, input_dataframe_with_signal_filename)
save_pickle(date_news_embedding, TRAINING_DIR, date_news_embedding_filename)
