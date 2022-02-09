# Code for [*Measure Estimation in the Barycentric Coding Model*](https://arxiv.org/abs/2201.12195)

## Downloading Data

To run the MNIST based experiments you will need to download the four files available from [here.](http://yann.lecun.com/exdb/mnist/)

For the NLP experiments you will need to download either the BBCSport or News20 files from [here.](https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0)

**You may need to modify the demo scripts to load in the datasets if you do not use the following file structure.**

```
code_folder
|  gaussian_demo.py 
|  mnist_demo.py
|  nlp_demo.py
|--utils
|  |  gaussian_utilities.py
|  |  mnist_utilities.py
|  |  nlp_utilities.py
|  |  opt_utilities.py
mnist
|  t10k-images-idx3-ubyte.gz
|  t10k-labels-idx1-ubyte.gz
|  train-images-idx3-ubyte.gz
|  train-labels-idx1-ubyte.gz
NLP
|  bbcsport_tr_te_split.mat
|  20ng2_500-emd-tr-te.mat
```

## Environment

To run the code you will need a `python` (this was developped with version 3.7) environment with the necessary packages installed. 

We recommend using [Conda](https://docs.conda.io/en/latest/) and the instructions below assume that you have this software already installed.

```
# create the environment
conda create --name ENVNAME python=3.7

# activate it
conda activate ENVNAME

# these are required for all demos
conda install -c conda-forge numpy scipy tqdm joblib pot cvxopt

# for the mnist demo you also need
conda install -c conda-forge mnist
```

For the Gaussian experiments, you will need PyTorch, and the installation depends on the machine that you are using. Follow the instructions on [this page](https://pytorch.org/get-started/locally/) for directions specific to your machine.

## Running Code

Assuming you've done everything above, you can run the following commands from the activate conda environment created above.

```
python gaussian_demo.py # used to make Figure 3
python mnist_demo.py    # used to make Figure 4
python nlp_demo.py      # used to make Figure 5
```