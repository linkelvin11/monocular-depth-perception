#! /bin/bash

if [ ! -e rgbd-dataset_full.tar ]
then
    #wget http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar
    wget http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/rgbd-dataset_full.tar
fi

if [ ! -d data ]
then
    mkdir -p data
    tar -xvf rgbd-dataset_full.tar --directory data
fi
