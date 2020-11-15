#!/bin/bash

PYTHON_RT=python3

export AZ_DATA=/dev/shm/az/

export PYTHONPATH=experiments/train/20x
$PYTHON_RT train.py

export PYTHONPATH=experiments/train/40x
$PYTHON_RT train.py

export PYTHONPATH=experiments/train/60x
$PYTHON_RT train.py
