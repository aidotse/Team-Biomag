#!/bin/bash
# File              : run_interactive_tensorflow2.0.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 23.10.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

ROOT_DIR=$PWD/..
DATA_DIR=$ROOT_DIR/data
CODE_DIR=$ROOT_DIR/src

nvidia-docker  run   \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/src \
	-it nvcr.io/nvidia/tensorflow:20.09-tf2-py3 \
	bash 




