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
deep prediction model training
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network.lms_data_generator import DataGenerator_average_torch
from network.util import load_pickle
from network.lms_cnn import *

# pre training paths/configs
resume_training = False
date_news_embedding_path = "training_dir/date_news_embedding" + "_" + EMBEDDING_METHOD + "_" + str(
	EMBEDDING_DIM) + ".pickle"
df_dt_adjclose_with_titles_path = "training_dir/df_dt_adjclose_with_titles.pickle"

model_name = "lms_cnn" + "_".join([str(i) for i in lsm_cnn_config_save_model])
model_save_path = TRAINING_DIR + "/" + model_name

# data preparation
date_news_embedding = load_pickle(date_news_embedding_path)
df_dt_adjclose_with_titles = load_pickle(df_dt_adjclose_with_titles_path)

datagenerator = DataGenerator_average_torch(df_dt_adjclose_with_titles, date_news_embedding,onehot_target=False)
train_loader, test_loader = datagenerator.prepare_dataset_torch(cuda=True,batch_size=BATCH_SIZE)

# init model and train
model = LMS_CNN_keras_wrapper()
# model.summary()
# if resume_training:
# 	model.load_weights(model_save_path)
model.fit(train_loader=train_loader,test_loader=test_loader,epochs=NUM_EPOCH)
# model.save(model_save_path)
