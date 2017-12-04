#!/usr/bin/env bash

FASTTEXT_ROOT=third_parties/fastText
EMBEDDING_SAVE_PATH=data/embedding/fasttext
MODEL_NAME=model_toy

./$FASTTEXT_ROOT/fasttext skipgram -input $1 -output $MODEL_NAME
mv $MODEL_NAME.vec $EMBEDDING_SAVE_PATH/$MODEL_NAME.vec
rm $MODEL_NAME.bin