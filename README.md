# News titles for stock prediction

Deep learning for stock long short signal prediction 
based on chronologically sorted news titles.

## Run 

### prepare or train your own embedding for the task

Trained word embedding based on news corpus provided from above dataset:
I have a trained word embedding based on the news corpus to get you started on 
the project under data/embedding/fasttext/model.vec


If you want to train your own word embedding:

```bash
sh scripts/prepare_fasttext.sh

# train the word embedding and save embedding under data/embedding, notice you might want to change model name
sh scripts/train_fasttext_embedding.sh data/train_own_word_embedding_example.txt 
```


Main code has three part:

run_preprocess.py: given input dataframe, prepare date_news_embeddings and input_dataframe_with_signal for training.
            vocabulary_processor and embedding matrix are saved for later use in online prediction. 

run_training.py: train long mid short term cnn with daily average news embedding. 

run_predicting.py:  online predictor for trained model. An english version toy example is given 
                  under run_predicting.py. After training without can mode_config.py, run this script get
                  a prediction result.

```python

# preprocess data
python run_preprocess.py

# traing model
python run_training.py

# online prediction 
python run_predicting.py
```

## Data

### Raw

1. news data

    original dataset news from Bloomberg \& Reuters from 20061020 to 20131126
    total number of trading days:1786
    [google drive](https://drive.google.com/open?id=0B3C8GEFwm08QY3AySmE2Z1daaUE)
    please refer to network.preprocess.get_data_dict to get raw text into structured
    format. 

2. market data 
     [SPY historical data from 20061020 to 20131126](https://finance.yahoo.com/quote/SPY/history?period1=1161273600&period2=1385395200&interval=1d&filter=history&frequency=1d)
    
### preprocessed data and data interface to the model

You can find raw data processed method under network.preprocess.
Feel free to try by yourself on cleaning up the raw data.

Now, let's focus on the data format sent to model pipeline. 
The interface to the model is a dataframe header as:
`Date,   Adj Close,  news_title`

An example row:<br>
 `2006-10-20  108.961449 [["news_title_1","00:00:00"],["news_title_2","01:00:00"]]`

In short we need date, adjust close price and _sorted_ daily news along reported time from old to new. Currently, the time
is only used for sorting within a day. 

You can find complete Bloomberg \& Reuters data interface example at training_dir/input_dataframe.pickle

## Model

### long mid short term CNN

![lms_cnn](img_src/lms_cnn.png)

input to long_mid_short cnn(or deep prediction model mentioned in the paper) has three parts:
of same essence<br>

events embedding: Most recent N days of events embedding with all events within a day represents
by one dense vector(currently averaged events embedding within a day). 

With long term use N = 30, mid term use N = 7, short term use N =1. 

Inputs on long term events embedding and mid term events embedding are Convolved and then Max pooled
. The output of both long and mid term from max pooling merged with short term input to form 
a dense vector represents a long_mid_short term embedding for the prediction on next day long 
or short signal. 


### Events Embedding

Event embedding can be generalized define as a dense vector represents an event.
(this implementation use word embeddings concatenation of news title sentence as event embedding)

Originally in paper[1], they follows thoughts in [3] and [4] trying to learn events embedding
based on triplet relations. 

Even though events embedding learned through method in [1] has around 3% ~ 4% improvements over
word embeddings concatenation of news title sentence as event embedding on the dataset in the paper.
It is not clear events embedding learned through Neural Tensor Network [3][4] can scale well at practise.
There are several issue considering events embedding learned through neural tensor network:

1. events triplet in the news title extraction can lose information: [1] use openIE and dependency parser 
to extract (Subject,Verb,Object) triplet assuming openIE extraction contains S,V,O and dependency parser
can help narrow words within the extractions of openIE to come up with the final SVO. 
However, the state-of-the-art openIE [5] can only extract 25%~50% of the news titles, leaving the rest of
SVO empty. This leads to huge information loss based on this method. 


2. events triplet in the news title extraction is tedious for agile project development:
given above description, to prepare the the event embedding, you need to extract SVO first based
on openIE and dependency parser, then you need to train the Neural Tensor Network to get the event 
embedding and save them to disk. Finally can you train CNN on event embeddings to get prediction 
input. For now it is not worth around 3% accuracy gain to prepare that much effort. 

3. events triplet in the news title extraction is not fully end to end for the project: It is impossible for the loss to be 
back-propagate to the event extraction part (aka neural tensor network), given our goal is to learn the next day long short signal.
The author believe if given enough data(currently only 1786 trading days news title), it is worth a deeper model to learn the
information extraction by a complete model itself instead of separate SVO extraction and prediction. 
 
![NTN](img_src/NTN.png)


## Experiments

experiments train/val results on average embedding based on word vector concatenation for news titles:
```
...
Epoch 411/1000
1404/1404 [==============================] - 1s 570us/step - loss: 0.6815 - acc: 0.6524 - val_loss: 0.7171 - val_acc: 0.6125
Epoch 412/1000
1404/1404 [==============================] - 1s 562us/step - loss: 0.6772 - acc: 0.6531 - val_loss: 0.7307 - val_acc: 0.6097
Epoch 413/1000
1404/1404 [==============================] - 1s 517us/step - loss: 0.6875 - acc: 0.6524 - val_loss: 0.7249 - val_acc: 0.6125
Epoch 414/1000
1404/1404 [==============================] - 1s 542us/step - loss: 0.6916 - acc: 0.6531 - val_loss: 0.7229 - val_acc: 0.6125
Epoch 415/1000
1404/1404 [==============================] - 1s 541us/step - loss: 0.6738 - acc: 0.6524 - val_loss: 0.7099 - val_acc: 0.6097
Epoch 416/1000
1404/1404 [==============================] - 1s 556us/step - loss: 0.6732 - acc: 0.6538 - val_loss: 0.7173 - val_acc: 0.6125
Epoch 417/1000
1404/1404 [==============================] - 1s 553us/step - loss: 0.6858 - acc: 0.6531 - val_loss: 0.7100 - val_acc: 0.6154
Epoch 418/1000
1404/1404 [==============================] - 1s 559us/step - loss: 0.6756 - acc: 0.6538 - val_loss: 0.7294 - val_acc: 0.6125
Epoch 419/1000
1404/1404 [==============================] - 1s 537us/step - loss: 0.6852 - acc: 0.6524 - val_loss: 0.7037 - val_acc: 0.6154
Epoch 420/1000
1404/1404 [==============================] - 1s 546us/step - loss: 0.6800 - acc: 0.6531 - val_loss: 0.7146 - val_acc: 0.6125
Epoch 421/1000
1404/1404 [==============================] - 1s 560us/step - loss: 0.6797 - acc: 0.6524 - val_loss: 0.7254 - val_acc: 0.6125
Epoch 422/1000
1404/1404 [==============================] - 1s 523us/step - loss: 0.6837 - acc: 0.6524 - val_loss: 0.7187 - val_acc: 0.6125
Epoch 423/1000
1404/1404 [==============================] - 1s 548us/step - loss: 0.6918 - acc: 0.6517 - val_loss: 0.7261 - val_acc: 0.6125
Epoch 424/1000
1404/1404 [==============================] - 1s 562us/step - loss: 0.6830 - acc: 0.6517 - val_loss: 0.7017 - val_acc: 0.6154
Epoch 425/1000
1404/1404 [==============================] - 1s 542us/step - loss: 0.6797 - acc: 0.6510 - val_loss: 0.7323 - val_acc: 0.6125
Epoch 426/1000
1404/1404 [==============================] - 1s 552us/step - loss: 0.6935 - acc: 0.6517 - val_loss: 0.7419 - val_acc: 0.6125
Epoch 427/1000
1404/1404 [==============================] - 1s 545us/step - loss: 0.6943 - acc: 0.6538 - val_loss: 0.7473 - val_acc: 0.6097
Epoch 428/1000
1404/1404 [==============================] - 1s 561us/step - loss: 0.6908 - acc: 0.6524 - val_loss: 0.7482 - val_acc: 0.6125
Epoch 429/1000
1404/1404 [==============================] - 1s 541us/step - loss: 0.6899 - acc: 0.6517 - val_loss: 0.7246 - val_acc: 0.6125
Epoch 430/1000
1404/1404 [==============================] - 1s 539us/step - loss: 0.6898 - acc: 0.6531 - val_loss: 0.7229 - val_acc: 0.6040
Epoch 431/1000
1404/1404 [==============================] - 1s 519us/step - loss: 0.6867 - acc: 0.6546 - val_loss: 0.7166 - val_acc: 0.6125
Epoch 432/1000
1404/1404 [==============================] - 1s 541us/step - loss: 0.6871 - acc: 0.6524 - val_loss: 0.7144 - val_acc: 0.6125
Epoch 433/1000
1404/1404 [==============================] - 1s 560us/step - loss: 0.6775 - acc: 0.6538 - val_loss: 0.7045 - val_acc: 0.6068
Epoch 434/1000
1404/1404 [==============================] - 1s 544us/step - loss: 0.6713 - acc: 0.6524 - val_loss: 0.7042 - val_acc: 0.6068
Epoch 435/1000
1404/1404 [==============================] - 1s 550us/step - loss: 0.6733 - acc: 0.6531 - val_loss: 0.7130 - val_acc: 0.6125
Epoch 436/1000
1404/1404 [==============================] - 1s 532us/step - loss: 0.6719 - acc: 0.6531 - val_loss: 0.7115 - val_acc: 0.6068
Epoch 437/1000
1404/1404 [==============================] - 1s 572us/step - loss: 0.6743 - acc: 0.6517 - val_loss: 0.7140 - val_acc: 0.6154
Epoch 438/1000
1404/1404 [==============================] - 1s 564us/step - loss: 0.6838 - acc: 0.6531 - val_loss: 0.7051 - val_acc: 0.6125
Epoch 439/1000
1404/1404 [==============================] - 1s 556us/step - loss: 0.6768 - acc: 0.6538 - val_loss: 0.7097 - val_acc: 0.6125
```

# Prepare Chinese Version input for the model 
The only difference between chinese and english in the sense of model input is the natural word segmentation on 
the english side. To prepare the experiments in Chinese, you need :<br>

## A chinese word segmenter

```bash
# get THULAC, compile and download model
sh scripts/prepare_chinese_word_segmenter.sh   

# A toy input example at data/chinese_word_seg_example.txt
sh scripts/run_chinese_word_segmenter.sh data/chinese_word_seg_example.txt
```

Now you have a segmented chinese corpus, you can train your own chinese word
embedding following the above instructions under train your own embedding.

## chinese counterpart input_dataframe.pickle

As mentioned above: the input_dataframe.pickle loads as a dataframe with header
`Date,   Adj Close,  news_title`

For chinese, you only need use chinese word segmenter to segment each chinese news
title with a space \' \'

An example row:<br>
 `2006-10-20  108.961449 [["将 句子 从 繁体 转化 为 简体","00:00:00"],["将 句子 从 繁体 转化 为 简体
","01:00:00"]]`


# references:

[1] [Ding, Xiao, Yue Zhang, Ting Liu and Junwen Duan. “Deep Learning for Event-Driven Stock Prediction.” IJCAI (2015).](https://www.semanticscholar.org/paper/Deep-Learning-for-Event-Driven-Stock-Prediction-Ding-Zhang/4938e8c8c9ea3d351d283181819af5e5801efbed)<br>
[2] [Bojanowski, Piotr, Edouard Grave, Armand Joulin and Tomas Mikolov. “Enriching Word Vectors with Subword Information.” TACL 5 (2017): 135-146.](https://www.semanticscholar.org/paper/Enriching-Word-Vectors-with-Subword-Information-Bojanowski-Grave/0a6383b13794452fb7339a7f8a5384885186ccf6)<br>
[3] [Socher, Richard, Danqi Chen, Christopher D. Manning and Andrew Y. Ng. “Reasoning With Neural Tensor Networks for Knowledge Base Completion.” NIPS (2013).](https://www.semanticscholar.org/paper/Reasoning-With-Neural-Tensor-Networks-for-Knowledg-Socher-Chen/50d53cc562225549457cbc782546bfbe1ac6f0cf)<br>
[4] [Mikolov, Tomas, Ilya Sutskever, Kai Chen, Gregory S. Corrado and Jeffrey Dean. “Distributed Representations of Words and Phrases and their Compositionality.” NIPS (2013).](https://www.semanticscholar.org/paper/Distributed-Representations-of-Words-and-Phrases-a-Mikolov-Sutskever/762b63d2eb86f8fd0de98a08561b77527ae8f165)<br>
[5] [OpenIE 5.0](https://github.com/dair-iitd/OpenIE-standalone)<br>
[6] [Zhongguo Li, Maosong Sun. Punctuation as Implicit Annotations for Chinese Word Segmentation. Computational Linguistics, vol. 35, no. 4, pp. 505-512, 2009.](https://github.com/thunlp/THULAC)<br>
