#!/bin/bash

WORKING_PATH=$PWD
LIBS_DIR=$WORKING_PATH/libs

mkdir -p $LIBS_DIR

# download pre-compiled HDF5 binary
wget https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/bin/linux-x86_64/hdf5-1.8.12-linux-x86_64-shared.tar.gz -O $LIBS_DIR/hdf5.tar.gz
mkdir -p $LIBS_DIR/hdf5
tar xzf $LIBS_DIR/hdf5.tar.gz -C $LIBS_DIR/hdf5 --strip-components=1
export HDF5_DIR=$LIBS_DIR/hdf5

# install required packages
pip install -r requirements.txt --user
