# Multi-View Data Generation Without View Supervision
An implementation of the models presented in the  [Multi-View Data Generation Without View Supervision](https://openreview.net/forum?id=ryRh0bb0Z) by Mickael Chen, Ludovic Denoyer and Thierry Artières, ICLR 2018

![gmv](https://github.com/mickaelChen/GMV/blob/master/GMV.png)

We propose a generative models for multi-view data by decomposing the latent space between content and view.

## Usage

The code runs using PyTorch and numpy.

Each file is a stand-alone for the training of one model.
```
python gmv.py
```
gmv and cgmv are proposed model.
gan2 is a simple baseline described in the paper.
mathieu is a pytorch reimplementation of [Disentangling factors of variation in deep representations using adversarial training](https://github.com/MichaelMathieu/factors-variation) using [DCGAN](https://arxiv.org/abs/1511.06434) inspired architecture.

Hyperparameters are set within the code and can be modified.
