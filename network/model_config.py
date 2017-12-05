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
config params for the network,training,testing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# preprocess
TRAINING_DIR = "training_dir"
MAX_SEQUENCE_LENGTH = 11
MIN_FREQ_FILTER = 2

MAX_NUM_WORDS = 50000
EMBEDDING_METHOD = 'fasttext'
EMBEDDING_DIM = 100

TRAIN_TEST_SEP_RATE = .8


# train base config
BATCH_SIZE = 256
NUM_EPOCH = 1000

## optimizer
LEARNING_RATE= 0.0005
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
DECAY = 0.01

## constrain
REGULARIZER_WEIGHT = 0.001

# lms_cnn_base
OUTPUT_DIM = 2

DENSE_HIDDEN_SIZE = 16
NEIGHBORHOOD_COMBINE = 3

SHORT_TERM_LENGTH = 1
MID_TERM_LENGTH = 7
LONG_TERM_LENGTH = 30

## lms_cnn
MID_TERM_CONV_KERNEL = (NEIGHBORHOOD_COMBINE, 20)
MID_TERM_POOL_SIZE = (1, 5)
LONG_TERM_CONV_KERNEL = (NEIGHBORHOOD_COMBINE, 80)
LONG_TERM_POOL_SIZE = (1, 20)

DENSE_HIDDEN_INPUT = int(EMBEDDING_DIM * MAX_SEQUENCE_LENGTH + \
	 (MID_TERM_LENGTH - NEIGHBORHOOD_COMBINE + 1)/MID_TERM_POOL_SIZE[0] * (EMBEDDING_DIM * MAX_SEQUENCE_LENGTH -MID_TERM_CONV_KERNEL[1])/MID_TERM_POOL_SIZE[1] + (LONG_TERM_LENGTH - NEIGHBORHOOD_COMBINE + 1)/LONG_TERM_POOL_SIZE[0] * (EMBEDDING_DIM * MAX_SEQUENCE_LENGTH -LONG_TERM_CONV_KERNEL[1])/LONG_TERM_POOL_SIZE[1])

# print DENSE_HIDDEN_INPUT

lms_cnn_base_config = [MAX_SEQUENCE_LENGTH, EMBEDDING_METHOD, EMBEDDING_DIM, BATCH_SIZE,
	NEIGHBORHOOD_COMBINE, LEARNING_RATE,REGULARIZER_WEIGHT]


lsm_cnn_config = lms_cnn_base_config +[DENSE_HIDDEN_SIZE, MID_TERM_CONV_KERNEL,MID_TERM_POOL_SIZE,
																						LONG_TERM_CONV_KERNEL, LONG_TERM_POOL_SIZE]

# this keeps config that must unchanged for the same model to be load
lsm_cnn_config_save_model = [MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
														NEIGHBORHOOD_COMBINE,
														DENSE_HIDDEN_SIZE, MID_TERM_CONV_KERNEL,MID_TERM_POOL_SIZE,
																						LONG_TERM_CONV_KERNEL, LONG_TERM_POOL_SIZE]
## lms_cnn3D
FILTER_SIZE = [2,3,4,5]

MID_TERM_CONV_KERNEL_3D = (NEIGHBORHOOD_COMBINE, NEIGHBORHOOD_COMBINE,EMBEDDING_DIM/6)
MID_TERM_CONV_KERNELS_3D = [(NEIGHBORHOOD_COMBINE, filter_size,EMBEDDING_DIM/6) for filter_size in FILTER_SIZE]

MID_TERM_3D_POOL_SIZE = (1, 3, 20)
MID_TERM_3D_POOLS_SIZE = [(1, MAX_SEQUENCE_LENGTH - filter_size, 2) for filter_size in FILTER_SIZE]

LONG_TERM_CONV_KERNEL_3D = (NEIGHBORHOOD_COMBINE, NEIGHBORHOOD_COMBINE,EMBEDDING_DIM/2)
LONG_TERM_CONV_KERNELS_3D = [(NEIGHBORHOOD_COMBINE, filter_size,EMBEDDING_DIM/2) for filter_size in FILTER_SIZE]

LONG_TERM_3D_POOL_SIZE = (1, 3, 20)
LONG_TERM_3D_POOLS_SIZE = [(1, MAX_SEQUENCE_LENGTH - filter_size, 3) for filter_size in FILTER_SIZE]

lsm_cnn3d_config = lms_cnn_base_config +[DENSE_HIDDEN_SIZE, FILTER_SIZE, MID_TERM_CONV_KERNEL, MID_TERM_3D_POOL_SIZE,
																							 LONG_TERM_CONV_KERNEL_3D, LONG_TERM_POOL_SIZE]

# this keeps config that must unchanged for the same model to be load
lsm_cnn3d_config_save_model = [MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
														NEIGHBORHOOD_COMBINE,
														DENSE_HIDDEN_SIZE, FILTER_SIZE, MID_TERM_CONV_KERNEL, MID_TERM_3D_POOL_SIZE,
														LONG_TERM_CONV_KERNEL_3D, LONG_TERM_POOL_SIZE]