#docker run -u --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
#    -v $PWD:/biomag -w /biomag \
#    bash
#    python -m train
# docker run -it --gpus all --rm -v /home/koosk/projects/adipocyte:/biomag -w /biomag tensorflow/tensorflow:latest-gpu python -c "print('hello')"
# docker run -it --gpus all --rm -v $PWD:/biomag -w /biomag tensorflow/tensorflow:latest-gpu bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass the data path in the first argument."
    exit
fi

docker run -it --gpus all --rm --name adipocyte -u $(id -u):$(id -g) -v $PWD:/biomag -v $1:/data biomag:adipocyte python train.py
