# !/usr/bin/bash
set -x
python3.6 inference.py\
        -model $1\
        -tgt_type $2\
        -device $3
