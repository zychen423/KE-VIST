#!/bin/bash
# bash run.sh 1 term rnn -global_att -self_att -concept_path

# Run following script
# bash run.sh 1 term rnn -self_att
set -x #echo on

python3.6 train.py \
        -log nos_$2_$3$4$5$6\
        -save_model trained\
        -save_mode best\
        -label_smoothing\
        -device $1\
        -batch_size 64\
        -tgt_type $2\
        -decoder $3\
        $4 $5 $6\
        -proj_share_weight

