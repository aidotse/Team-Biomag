#!/bin/bash
# File              : run_interactive_pytorch.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 23.10.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>


nvidia-docker  run   \
       -v $DATA_DIR:/data \
       -v $CODE_DIR:/src \
       -it nvcr.io/nvidia/pytorch:20.09-py3 \
       bash
