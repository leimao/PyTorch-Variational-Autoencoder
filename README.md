# PyTorch Variational Autoencoder

## Introduction

## Usages

### Build Docker Image

```bash
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:2.2.0 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/mnt pytorch:2.2.0
```

### Train Variational Autoencoder

```bash
$ python train.py
```

### Examine Results

The results will be saved to the `results` directory.

## References

- [Variational Autoencoder](https://leimao.github.io/blog/blog/Variational-Autoencoder/)
- [PyTorch Variational Autoencoder](https://leimao.github.io/blog/blog/PyTorch-Variational-Autoencoder/)
