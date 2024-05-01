# PyTorch Variational Autoencoder

## Introduction

## Usages

### Build Docker Image

```
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:2.2.0 .
```

### Run Docker Container

```
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch:2.2.0
```

## Usages

### PyTorch Official Example

https://botorch.org/tutorials/vae_mnist
https://github.com/pytorch/examples/tree/main/vae

This is an improved implementation of the paper [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

```bash
pip install -r requirements.txt
python main.py
```

The main.py script accepts the following arguments:

```bash
optional arguments:
  --batch-size		input batch size for training (default: 128)
  --epochs		number of epochs to train (default: 10)
  --no-cuda		enables CUDA training
  --mps         enables GPU on macOS
  --seed		random seed (default: 1)
  --log-interval	how many batches to wait before logging training status
```
