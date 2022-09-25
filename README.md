# ESGNN

Code for ["Echo state graph neural networks with analogue random resistor arrays."](https://arxiv.org/abs/2112.15270)

[![arXiv](https://img.shields.io/badge/arXiv-2112.15270-b31b1b.svg)](https://arxiv.org/abs/2112.15270) ![License](https://img.shields.io/badge/license-MIT-yellow) [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

## Abstract

Recent years have witnessed an unprecedented surge of interest, from social networks to drug discovery, in learning representations of graph-structured data. However, graph neural networks, the machine learning models for handling graph-structured data, face significant challenges when running on conventional digital hardware, including von Neumann bottleneck incurred by physically separated memory and processing units, slowdown of Moore’s law due to transistor scaling limit, and expensive training cost. Here we present a novel hardware-software co-design, the random resistor array-based echo state graph neural network, which addresses these challenges. The random resistor arrays not only harness low-cost, nanoscale and stackable resistors for highly efficient in-memory computing using simple physical laws, but also leverage the intrinsic stochasticity of dielectric breakdown to implement random projections in hardware for an echo state network that effectively minimizes the training complexity thanks to its fixed and random weights. The system demonstrates state-of-the-art performance on both graph classification using the MUTAG and COLLAB datasets and node classification using the CORA dataset, achieving 38.71×, 95.04×, and 579.66× improvement of energy efficiency over digital hardware and 99.35%, 99.99%, and 91.40% reduction of backward pass complexity compared to conventional graph learning, respectively, which may pave the way for the next generation AI system for graph learning.

## Requirements

The codes are tested on Ubuntu 20.04, CUDA 11.2 with the following packages:

```shell
torch == 1.9.0
torch-geometric == 1.7.2
scipy == 1.7.0
numpy == 1.20.2
```

**Note:** The graph classification tasks can be executed without torch-geometric (PyG). 

## Installation

You can install the required dependencies with the following code.

```shell
conda create -n ESGNN python=3.8
conda activate ESGNN
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge --yes
CUDA=cu111
TORCH=1.9.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html 
pip install torch-geometric==1.7.2 
```

## Demo

For graph classification on `MUTAG` dataset, run the following line in terminal:

```shell
bash run_mutag.sh
```

For graph classification on `COLLAB` dataset, run the following line in terminal:

```shell
bash run_collab.sh
```

For node classification on `Cora` dataset, run the following line in terminal:

```shell
bash run_cora.sh
```

**Note:** The code for graph classification simulation provided in this demo is the same with that using the random resistive memory hardware. The only difference is the weight multiplication function `WeightMultiplication` performs hardware calls to a Xilinx FPGA via [`pynq.dma`](https://pynq.readthedocs.io/en/v2.5/pynq_libraries/dma.html).

## Dataset

Both the processed `MUTAG` and `COLLAB` datasets are provided in `data` folder. The raw data can be downloaded [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

The `Cora` dataset can be automatically downloaded in the code via PyG. The raw data can also be downloaded [here](https://relational.fit.cvut.cz/dataset/CORA).
