#!/usr/bin/env bash

mkdir -p third_parties
cd third_parties
git clone https://github.com/facebookresearch/fastText.git

cd fastText
make