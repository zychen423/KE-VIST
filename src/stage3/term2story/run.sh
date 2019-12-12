#!/bin/bash
set -x
python train.py -log $2 -save_model trained -save_mode best -label_smoothing -device $1 -positional $3
