if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass the data path in the first argument."
    exit
fi

docker run -it --gpus all --rm --name adipocyte -u $(id -u):$(id -g) -v $PWD:/biomag -v $1:/data biomag:adipocyte python train.py