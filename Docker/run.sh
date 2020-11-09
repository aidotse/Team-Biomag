#!/bin/bash
# File              : run.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 23.10.2020
# Last Modified Date: 23.10.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

ROOT_DIR=$PWD/..
DATA_DIR=$ROOT_DIR/../data
CODE_DIR=$ROOT_DIR/src

nvidia-docker  run   \
	-v $DATA_DIR:/data \
	-v $CODE_DIR:/src \
	-it biomag:adipocyte \
	--name biomag_container \
	/bin/bash 



