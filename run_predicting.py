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
deep prediction model online version for predicting
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from network.lms_cnn import *
from network.util import load_pickle
from network.preprocess import clean_str
import nlp_vocab


class LMS_Predictor:
	"""
	long_mid_short_term_cnn online predictor
	"""
	def __init__(self,model_name=None,vocab_processor_path=None, embedding_matrix_path=None):
		"""
		init the predictor
		:type vocab_processor_path: str	| path to vocab_processor_path
		:type embedding_matrix_path: str| path to embedding_matrix_path
		:type model_name: str| model_name that is consistent with model_name
						defination in network/model_config lsm_cnn_config_save_model

		"""
		if model_name:
			self.model_name = model_name
		else: # default unchanged save_model params
			self.model_name = "lms_cnn" + "_".join([str(i) for i in lsm_cnn_config_save_model])

		if not vocab_processor_path:
			vocab_processor_path = TRAINING_DIR + "/" + "vocab_processor"
		self.vocab_processor = nlp_vocab.VocabularyProcessor.restore(vocab_processor_path)

		if not embedding_matrix_path:
			embedding_matrix_path = TRAINING_DIR + "/" + "embedding_matrix.pickle"
		self.embedding_matrix = load_pickle(embedding_matrix_path)

		saved_model_path = TRAINING_DIR + "/" + self.model_name
		self.model = build_deep_prediction_model()
		self.model.load_weights(saved_model_path)

	def _predict(self, X):
		"""
		predict long, short given long, mid, short term vectors.
		:param X:dict| dict that has long, mid, short term vectors,
				for example:
				X = {"short_term": short_term_test, "mid_term": mid_term_test, "long_term": long_term_test}

		:return: list|prediction result with 0 indicates short, 1 indicates long.
		"""
		short_term_test = X[0].reshape(SHORT_TERM_LENGTH,MAX_SEQUENCE_LENGTH*EMBEDDING_DIM)
		mid_term_test = X[:MID_TERM_LENGTH].reshape(1,MID_TERM_LENGTH,MAX_SEQUENCE_LENGTH*EMBEDDING_DIM)
		long_term_test = X.reshape(1,LONG_TERM_LENGTH,MAX_SEQUENCE_LENGTH*EMBEDDING_DIM)

		input_dict = {"short_term": short_term_test, "mid_term": mid_term_test, "long_term": long_term_test}

		result = self.model.predict(input_dict)

		return [np.argmax(r)for r in result]

	def predict(self, texts):
		"""
		predict on raw news text with long term input (30 days). with lower index refers
		to close to present dates, so 0 points to today's news given you want to predict
		tomorrow.

		:param texts:list of list of str|
		:return: text_vectors | np.array, with each date has one vector representation
		"""
		assert len(texts) == LONG_TERM_LENGTH

		date_news_embedding = []

		for date_all_news in texts:
			x_text_cleaned = [clean_str(x) for x in date_all_news]
			x_text_cleaned_transformed = self.vocab_processor.transform(x_text_cleaned)

			sentence_embeddings = [[self.embedding_matrix[i] for i in ids] for ids in
														 	x_text_cleaned_transformed]

			date_embedding = np.mean(sentence_embeddings, axis=0)
			date_news_embedding.append(date_embedding)

		return self._predict(np.array(date_news_embedding))

# toy example of use online predictor
# ===================================
# For online testing, usually we only has one long term
# (30 trading days) period to test the data.

toy_news_texts = [["Assad Promises to Have Syria Under Control Soon, Chavez says."],
["Chavez Says He Will Return to Cuba Late Saturday."],
["No timetable for restarting California nuclear plant: Jaczko."],
["Japan Approves Rules for Nuclear Restarts, Yomiuri Says."],
["Employment Increase in U.S. Trails Most-Pessimistic Forecasts."],
["Treasuries Gain After Jobs Increase Falls Below Forecasts."],
["Bernanke Warning on Jobs Vindicated by March Payrolls Report."],
["DMS Loses Challenge to $32 Billion Contract Held by McKesson."],
["Euro Falls Most in 11 Months Versus Yen on Debt Concern."],
["Running Back Brandon Jacobs Moves to 49ers from New York Giants."],
["Rangers’ Tortorella Fined by NHL After Calling Penguins Arrogant."],
["Canadian Dollar Posts First Weekly Rise in Month on Job Gains."],
["Japan, China to ‘Consult Closely’ on Support for IMF, Azumi Says."],
["Reforming Myanmar targets boost in rice exports."],
["Spain Will Manage Without External Help, De Guindos Tells FAZ."],
["Saudi Arabia Uses 600,000 Barrels a Day Oil Products, Watan Says."],
["High-End Homebuyers Find Stamp Duty Tax Loopholes, FT Says."],
["Almarai Co., Jarir Marketing: Saudi Arabia Equity Preview."],
["Pakistan, Egypt Lead Kenyan Tea Buying in First 2 Months of Year."],
["U.S. Nominates Mitchell as Ambassador to Myanmar, FT Says."],
["FROCH ENTERPRISE March Sales Fall 18.68% (Table) : 2030 TT."],
["Gazprom, BASF, E.ON Tap Siberia Turonian-Age Gas Field, RIA Says."],
["India's jewelers call off strike."],
["Swiss Life Says Insurance Acquisitions Not Easy, F&W Reports."],
["Russia Plans Moon, Venus, Mercury Exploration Projects, RIA Says."],
["Coty Chairman Becht Steps Up Drive to Acquire Avon, FT Reports."],
["Jarir Marketing Poised for Highest Close on Record After Profit."],
["EDF to Chase Offshore Wind Orders Abroad, Proglio Tells Figaro."],
["Jobs recovery suffers setback in March."],
["Saudi Arabia Equity Movers: Almarai, Jarir Marketing and Spimaco.",
"Orange, Thales to Get French Cloud Computing Funds, Figaro Says.",
"Stansted Could Double Passengers on Deregulation, Times Reports."]]

# init model and predict
print ("Loading Trained Model")
model = LMS_Predictor()
result = model.predict(toy_news_texts)
print (result)

