#!/usr/bin/env bash
# 注意均为UTF8文本

./third_parties/THULAC/thulac -t2s -seg_only -deli ' ' -input $1 -output $1_output.txt -model_dir third_parties/THULAC/models