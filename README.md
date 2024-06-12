# PyTorch Variational Autoencoder

## Introduction

PyTorch variational autoencoder (VAE) example for MNIST dataset. The modeled posterior distribution follows a Gaussian distribution with a full covariance matrix.

## Usages

### Build Docker Image

```bash
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:2.2.0 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch:2.2.0
```

### Run Variational Autoencoder MNIST Training

```bash
$ python train.py
```

### Examine Results

The results will be saved to the `results` directory.

## References

- [Variational Autoencoder](https://leimao.github.io/blog/Variational-Autoencoder/)
- [PyTorch Variational Autoencoder](https://leimao.github.io/blog/PyTorch-Variational-Autoencoder/)
