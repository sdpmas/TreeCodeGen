#!/bin/bash

test_file="data/conala/test.bin"
#load your model here
model_file=''
python exp.py \
    --cuda \
    --mode test \
    --beam_size 30 \
    --load_model ${model_file} \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --decode_max_time_step 100

