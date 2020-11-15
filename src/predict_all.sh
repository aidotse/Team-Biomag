#!/bin/bash

PYTHON_RT=python3

export AZ_DATA=/dev/shm/az/

export PYTHONPATH=experiments/prediction/20x
$PYTHON_RT train.py

export PYTHONPATH=experiments/prediction/40x
$PYTHON_RT train.py

export PYTHONPATH=experiments/prediction/60x
$PYTHON_RT train.py
