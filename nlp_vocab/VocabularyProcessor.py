# -*- coding: utf-8 -*-
# Copyright 2017 The Wenchen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tokenizer for processing the corpus

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
from nlp_vocab.Vocab import Vocabulary

try:
	import cPickle as pickle
except:
	import pickle

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
													re.UNICODE)


def tokenizer(iterator):
	"""Tokenizer generator.
		Args:
			iterator: Input iterator with strings.

		Yields:
			array of tokens per each value in the input.
	"""
	for value in iterator:
		yield TOKENIZER_RE.findall(value.lower())


class VocabularyProcessor(object):
	"""
	VocabularyProcessor helps build Vocabulary given the corpus to be fitted.
	If documents are longer, they will be trimmed, if shorter - padded.
	default special tokens for the vocabulary are __UNK__ and __PAD__.
	"""

	def __init__(self,
							 max_document_length,
							 min_frequency=0,
							 vocabulary=None,
							 tokenizer_fn=None):
		"""Initializes a VocabularyProcessor instance.
		Args:
			max_document_length: Maximum length of documents.
			if documents are longer, they will be trimmed, if shorter - padded.
			min_frequency: Minimum frequency of words in the vocabulary.
			vocabulary: CategoricalVocabulary object.
		Attributes:
			vocabulary_: Vocabulary object.
		"""
		self.max_document_length = max_document_length
		self.min_frequency = min_frequency
		if vocabulary:
			self.vocabulary_ = vocabulary
		else:
			self.vocabulary_ = Vocabulary()
		if tokenizer_fn:
			self._tokenizer = tokenizer_fn
		else:
			self._tokenizer = tokenizer

	def fit(self, raw_documents):
		"""Learn a vocabulary dictionary of all tokens in the raw documents.

		Args:
			raw_documents: An iterable which yield either str or unicode.

		Returns:
			self
		"""
		for tokens in self._tokenizer(raw_documents):
			for token in tokens:
				self.vocabulary_.add(token)
		if self.min_frequency > 0:
			self.vocabulary_.trim(self.min_frequency)
		self.vocabulary_.freeze()
		return self

	def fit_transform(self, raw_documents):
		"""Learn the vocabulary dictionary and return indexies of words.
			Args:
				raw_documents: An iterable which yield either str or unicode.

			Returns:
				x: iterable, [n_samples, max_document_length]. Word-id matrix.
		"""
		self.fit(raw_documents)
		return self.transform(raw_documents)

	def transform(self, raw_documents):
		"""Transform documents to word-id matrix.
			Convert words to ids with vocabulary fitted with fit or the one
			provided in the constructor.

			Args:
				raw_documents: An iterable which yield either str or unicode.

			Yields:
				x: iterable, [n_samples, max_document_length]. Word-id matrix.
		"""
		for tokens in self._tokenizer(raw_documents):
			word_ids = np.ones(self.max_document_length, np.int64)  # default addtional missing slot use pad
			for idx, token in enumerate(tokens):
				if idx >= self.max_document_length:
					break  # cut exceeding part, this means user,given their special tokens, need to add their own
				# special tokens by themself before
				word_ids[idx] = self.vocabulary_.get(token)
			yield word_ids

	def reverse(self, documents):
		"""Reverses output of vocabulary mapping to words.

			Args:
				documents: iterable, list of class ids.

			Yields:
				Iterator over mapped in words documents.
		"""
		for item in documents:
			output = []
			for class_id in item:
				output.append(self.vocabulary_.reverse(class_id))
			yield ' '.join(output)

	def prepare_embedding_matrix(self, embedding_filepath):
		"""
		after fit the doc, prepare the corresponding embedding
		matrix of which the index is consistent with word index.
		:param embedding_filepath|str: embedding file path, embedding could be either binary or text
		:return: embedding_matrix|np.array, embedding_matrix which vector index is consistent
																with word index in vocab after trim
		"""
		def check_emb_dim():
			with open(embedding_filepath, 'r') as fin:  # TODO binary load
				for i, value in enumerate(fin):
					if i == 0: continue  # might be header
					values = np.asarray(value.split()[1:], dtype='float32')
					return values.shape[0]

		assert self.vocabulary_.get_freeze_state() == True, \
			"vocabulary should be freezed before prepare embedding"

		num_words = self.vocabulary_.__len__()
		embedding_dim = check_emb_dim()
		embedding_matrix = np.zeros((num_words, embedding_dim))

		# load and add necessary embedding
		with open(embedding_filepath, 'r') as f:  # TODO binary load
			for line in f:
				values = line.split()
				word = values[:len(values)-embedding_dim]
				word = " ".join(word)
				word_index = self.vocabulary_.get(word)
				if word_index != self.vocabulary_.UNK_id:  # if embedding words in vocab, load
					vector = np.asarray(values[-embedding_dim:], dtype='float32')
					embedding_matrix[word_index] = vector

		return embedding_matrix

	def save(self, filename):
		"""Saves vocabulary processor into given file.
			Args:
			filename: Path to output file.
		"""
		with open(filename, 'wb') as f:
			f.write(pickle.dumps(self))

	@classmethod
	def restore(cls, filename):
		"""Restores vocabulary processor from given file.
			Args:
				filename: Path to file to load from.

			Returns:
				VocabularyProcessor object.
		"""
		with open(filename, 'rb') as f:
			return pickle.loads(f.read())