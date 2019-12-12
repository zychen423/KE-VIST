#!/bin/bash
set -x
python train.py -log $2 -save_model trained -save_mode all -label_smoothing -device $3 -model $1 -vist -positional $4
