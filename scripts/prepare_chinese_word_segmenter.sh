#!/usr/bin/env bash

mkdir -p third_parties
cd third_parties
git clone https://github.com/thunlp/THULAC.git

cd THULAC
make

wget http://thulac.thunlp.org/source/Models_v1_v2.zip
unzip Models_v1_v2.zip