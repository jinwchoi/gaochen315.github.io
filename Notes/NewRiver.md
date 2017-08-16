# Setup Cluster

[General Information](https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples)
[Group Web](https://mlp.ece.vt.edu/wiki/doku.php)

## Connect to Cluster
If off-campus, need to use port 2222
`ssh -p 2222 chengao@fukushima.ece.vt.edu`
`ssh -p 2222 chengao@godel.ece.vt.edu`

## Setting password-less SSH
From personal computer:
```
scp -r <userid>@godel.ece.vt.edu:/srv/share/lab_helpful_files/ ~/
```
Change ``<userid>`` to your CVL account username in the ~/lab_helpful_files/config file and move it to ~/.ssh
```
mv ~/lab_helpful_files/config ~/.ssh/
ssh-keygen -t rsa
```
Enter the password here. Make sure ~/ on sever has .ssh folder: login, does

```
cd ~/.ssh
```
work? if not, type
```
mkdir .ssh
scp ~/.ssh/id_rsa.pub godel:~/.ssh/
```
On server:

```
cd ~/.ssh/
cat id_rsa.pub >> authorized_keys2
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys2
```
Now you should be able to type something like ``$ ssh huck`` On your personal computer and it will login without asking for a password.

## Screen

Open a screen
```
screen
```
Check running jobs
```
screen -list
```
Run whatever jobs you want. If run matlab,
```
matlab -nodisplay -r "Rich_new;exit"
```

Check your running jobs
```
screen -r 899
```
Exit
```
Press control + A, then press D
```

## Copy
```
rsync --partial --progress --rsh=ssh chengao@godel.ece.vt.edu:/home/chengao/BIrdDetection/Chen_code/50mm_21May2013* ./
```


## Jupyter Notebook

1. In a terminal. Connect to server. Ask for GPU.
`srun -w fukushima --gres gpu:k80:2 --pty bash`
Ask for environment
`source activate TF`
Open Jupyter Notebook
`jupyter notebook`

2. Open a new terminal, type
`ssh -N -L localhost:8888:localhost:8888chengao@fukushima.ece.vt.edu`
and type password. First 8888 is the local port which we need at step 3. Second 8888 is the port we get from step 1.

3. Open Chrome, type
`http://localhost:8888/`

## Preparation

### Install miniconda

#### NewRiver(Linux)
```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
```
#### Huck(PPC64)
```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-ppc64le.sh
chmod +x Miniconda2-latest-Linux-ppc64le.sh
./Miniconda2-latest-Linux-ppc64le.sh
```
- Create an environment named TF (or whatever name you want)
```
source ~/.bashrc
conda create -n TF python
```
### Install dependency
```
/home/chengao/miniconda2/envs/TF/bin/pip install numpy
```
or
```
conda install numpy
```
Update .bashrc after install some library


### Install Caffe

  - config
```
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#       You should not set this flag if you will be reading LMDBs with any
#       possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
#OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda-8.0
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_52,code=sm_52 \
                -gencode arch=compute_60,code=sm_60 \
                -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_61,code=compute_61
# Deprecated
# CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
#               -gencode arch=compute_20,code=sm_21 \
#               -gencode arch=compute_30,code=sm_30 \
#               -gencode arch=compute_35,code=sm_35 \
#               -gencode arch=compute_50,code=sm_50 \
#               -gencode arch=compute_52,code=sm_52 \
#               -gencode arch=compute_60,code=sm_60 \
#               -gencode arch=compute_61,code=sm_61 \
#               -gencode arch=compute_61,code=compute_61

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
#PYTHON_INCLUDE := /usr/include/python2.7 \
                /usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
ANACONDA_HOME := $(HOME)/miniconda2/envs/TF
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                $(ANACONDA_HOME)/include/python2.7 \
                $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
#PYTHON_LIB := /usr/lib /usr/local/lib
PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /home/chengao/cudnn/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib64/atlas /home/chengao/cudnn/lib64

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# NCCL acceleration switch (uncomment to build with NCCL)
# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
# USE_NCCL := 1

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @                                                 

```

- Install Tensorflow


### Toolbox
- Install OpenPose

### Dataset
- V-COCO
 Train a RCNN on COCO 2014_train Dataset
 ```
 python ./tools/train_net.py --device cpu   --weights data/pretrain_model/VGG_imagenet.npy   --imdb coco_2014_train   --iters 1   --cfg experiments/cfgs/faster_rcnn_end2end.yml   --network VGGnet_train
 ```
 Test a RCNN on COCO 2014_val Dataset

 ```
python ./tools/test_net.py --device gpu --weights output/faster_rcnn_end2end/coco_2014_train/VGGnet_fast_rcnn_iter_90000.ckpt --imdb coco_2014_val --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_test
  ```

## How to debug
```
try :
    xxx
except:
    import pdb
    pdb.set_trace()
```

## How to submit jobs on New River

1. Write a shell script for submission of jobs on NewRiver. You may only modify "#PBS -l" section for resource request. And state what function you want to run at the bottom of the script.
```
#!/bin/bash
#
# Annotated example for submission of jobs on NewRiver
#
# Syntax
# '#' denotes a comment
# '#PBS' denotes a PBS directive that is applied during execution
#
# More info
# https://secure.hosting.vt.edu/www.arc.vt.edu/computing/newriver/#examples
#
# Chen Gao
# Aug 16, 2017
#

# Account under which to run the job
#PBS -A vllab_2017

# Access group. Do not change this line.
#PBS -W group_list=newriver

# Set some system parameters (Resource Request)
#
# NewRiver has the following hardware:
#   a. 100 24-core, 128 GB Intel Haswell nodes
#   b.  16 24-core, 512 GB Intel Haswell nodes
#   c.   8 24-core, 512 GB Intel Haswell nodes with 1 Nvidia K80 GPU
#   d.   2 60-core,   3 TB Intel Ivy Bridge nodes
#   e.  39 28-core, 512 GB Intel Broadwell nodes with 2 Nvidia P100 GPU
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 2 nodes with 24 cores each
#   #PBS -l nodes=1:ppn=24
#   Request 4 cores (on any number of nodes)
#   #PBS -l procs=4
#   Request 12 cores with 20gb memory per core
# 	#PBS -l procs=12,pmem=20gb
#   Request 2 nodes with 24 cores each and 20gb memory per core (will give two 512gb nodes)
#   #PBS -l nodes=2:ppn=24,pmem=20gb
#   Request 2 nodes with 24 cores per node and 1 gpu per node
#   #PBS -l nodes=2:ppn=24:gpus=1
#   Request 2 cores with 1 gpu each
#   #PBS -l procs=2,gpus=1
#PBS -l procs=12,pmem=16gb,walltime=2:20:00:00

# Set Queue name
#   normal_q        for production jobs on all Haswell nodes (nr003-nr126)
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers (nr001-nr002)
#   dev_q           for development/debugging jobs on Haswell nodes. These jobs must be short but can be large.
#   vis_q           for visualization jobs on K80 GPU nodes (nr019-nr027). These jobs must be both short and small.
#   open_q          for jobs not requiring an allocation. These jobs must be both short and small.
#   p100_normal_q   for production jobs on P100 GPU nodes
#   p100_dev_q      for development/debugging jobs on P100 GPU nodes. These jobs must be short but can be large.
# For more on queues as policies, see http://www.arc.vt.edu/newriver#policy
#PBS -q normal_q

# Send emails to -M when
# a : a job aborts
# b : a job begins
# e : a job ends
#PBS -M <PID>@vt.edu
#PBS -m bea

# Add any modules you might require. This example adds matlab module.
# Use 'module avail' command to see a list of available modules.
#
module load matlab

# Navigate to the directory from which this script was executed
cd /home/chengao/BIrdDetection/Chen_code

# Below here enter the commands to start your job. A few examples are provided below.
# Some useful variables set by the job:
#  $PBS_O_WORKDIR    Directory from which the job was submitted
#  $PBS_NODEFILE     File containing list of cores available to the job
#  $PBS_GPUFILE      File containing list of GPUs available to the job
#  $PBS_JOBID        Job ID (e.g., 107619.master.cluster)
#  $PBS_NP           Number of cores allocated to the job

### If run Matlab job ###
#
# Open a MATLAB instance and call Rich_new()
#matlab -nodisplay -r "addpath('Chen_code'); Rich_new;exit"

### If run Tensorflow job ###
#
```

2. To submit your job to the queuing system, use the command qsub. For example, if your script is in "JobScript.qsub", the command would be:
```
qsub ./JobScript.qsub
```
3. This will return your job name of the form. 230077 is the job number
```
230077.master.cluster
```
4. To check a job’s status, use the checkjob command:
```
checkjob -v 230077
```
5. To check resource usage on the nodes available to a running job, use:
```
jobload 230077
```
6. To remove a job from the queue, or stop a running job, use the command qdel
```
qdel 230077
```
