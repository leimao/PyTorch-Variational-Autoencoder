import math
import os
import random
import statistics

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class VariationalEncoder(nn.Module):

    def __init__(self,
                 num_observed_dims=784,
                 num_latent_dims=8,
                 num_hidden_dims=512):
        super(VariationalEncoder, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.fc = nn.Linear(in_features=self.num_observed_dims,
                            out_features=self.num_hidden_dims)
        # Encoded variational mean.
        self.fc_mean = nn.Linear(in_features=self.num_hidden_dims,
                                 out_features=self.num_latent_dims)
        # Encoded variational log standard deviation.
        self.fc_log_std = nn.Linear(in_features=self.num_hidden_dims,
                                    out_features=self.num_latent_dims)
        # Encoded flattened unmasked lower triangular matrix.
        self.fc_unmasked_lower_triangular_flatten = nn.Linear(
            in_features=self.num_hidden_dims,
            out_features=self.num_latent_dims * self.num_latent_dims)
        # Constant mask for lower triangular matrix.
        self.mask = torch.tril(torch.ones(self.num_latent_dims,
                                          self.num_latent_dims),
                               diagonal=-1)
        self.register_buffer('mask_const', self.mask)

        # Using MultivariateNormal for sampling is awkward in PyTorch as of PyTorch 2.2,
        # because it always produces samples on CPU.
        # from torch.distributions.multivariate_normal import MultivariateNormal
        # self.std_normal_mu = torch.zeros(self.num_latent_dims)
        # self.std_normal_std = torch.eye(self.num_latent_dims)
        # self.register_buffer('std_normal_mu_const', self.std_normal_mu)
        # self.register_buffer('std_normal_std_const', self.std_normal_std)
        # self.multivariate_std_normal = MultivariateNormal(self.std_normal_mu_const, self.std_normal_std_const)

    def encode(self, x):

        h = torch.relu(self.fc(x))
        mu = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        unmasked_lower_triangular_flatten = self.fc_unmasked_lower_triangular_flatten(
            h)
        unmasked_lower_triangular = unmasked_lower_triangular_flatten.view(
            -1, self.num_latent_dims, self.num_latent_dims)

        return mu, log_std, unmasked_lower_triangular

    def reparameterize(self, mu, log_std, unmasked_lower_triangular):

        # Perform one sampling operation for each sample in the batch.
        # Using full-covariance Gaussian posterior
        std = torch.exp(log_std)
        # torch.diag_embed diagonalizes the vector in batches.
        lower_triangular = unmasked_lower_triangular * self.mask_const + torch.diag_embed(
            std)
        # Sample from standard normal in batch.
        # eps = self.multivariate_std_normal.sample(sample_shape=torch.Size([mu.shape[0]]))
        # The variables in the multivariate standard distribution are independent and follows the univariate standard normal distribution.
        # Thus we can use the following trick to sample from the multivariate standard normal distribution.
        eps = torch.randn_like(std)
        z = mu + torch.bmm(lower_triangular,
                           eps.view(-1, self.num_latent_dims, 1)).view(
                               -1, self.num_latent_dims)

        return z, eps

    def forward(self, x):

        mu, log_std, unmasked_lower_triangular = self.encode(x)
        z, eps = self.reparameterize(mu, log_std, unmasked_lower_triangular)

        return z, eps, log_std


class Decoder(nn.Module):

    def __init__(self,
                 num_observed_dims=784,
                 num_latent_dims=8,
                 num_hidden_dims=512):
        super(Decoder, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.fc = nn.Linear(in_features=self.num_latent_dims,
                            out_features=self.num_hidden_dims)
        self.fc_out = nn.Linear(in_features=self.num_hidden_dims,
                                out_features=self.num_observed_dims)

    def decode(self, z):

        h = torch.relu(self.fc(z))
        x = torch.sigmoid(self.fc_out(h))

        return x

    def forward(self, z):

        return self.decode(z)


class VAE(nn.Module):

    def __init__(self,
                 num_observed_dims=784,
                 num_latent_dims=8,
                 num_hidden_dims=512):
        super(VAE, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.encoder = VariationalEncoder(
            num_observed_dims=self.num_observed_dims,
            num_latent_dims=self.num_latent_dims,
            num_hidden_dims=self.num_hidden_dims)
        self.decoder = Decoder(num_observed_dims=self.num_observed_dims,
                               num_latent_dims=self.num_latent_dims,
                               num_hidden_dims=self.num_hidden_dims)

    def forward(self, x):

        z, eps, log_std = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, eps, log_std


def compute_negative_evidence_lower_bound(x, x_reconstructed, z, eps, log_std):

    pi = torch.tensor(math.pi).to(x.device)

    # Reconstruction loss.
    # E[log p(x|z)]
    log_px = -torch.nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction="sum")
    # E[log q(z|x)]
    log_qz = -0.5 * torch.sum(eps**2 + log_std + torch.log(2 * pi))
    # E[log p(z)]
    # Assuming standard normal prior.
    log_pz = -0.5 * torch.sum(z**2 + torch.log(2 * pi))
    # ELBO
    elbo = log_px + log_pz - log_qz
    negative_elbo = -elbo

    # Compute average negative ELBO.
    batch_size = x.shape[0]
    negative_elbo_avg = negative_elbo / batch_size

    return negative_elbo_avg


class BinarizeTransform(object):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold).float()


def prepare_cifar10_dataset(root="data"):

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # Binarize the input using some threshold.
        # This will improve the performance of the model.
        BinarizeTransform(threshold=0.5),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        BinarizeTransform(threshold=0.5),
    ])

    train_set = torchvision.datasets.MNIST(root="data",
                                           train=True,
                                           download=True,
                                           transform=train_transform)

    test_set = torchvision.datasets.MNIST(root="data",
                                          train=False,
                                          download=True,
                                          transform=test_transform)

    class_names = train_set.classes

    return train_set, test_set, class_names


def prepare_cifar10_dataloader(train_set,
                               test_set,
                               train_batch_size=128,
                               eval_batch_size=256,
                               num_workers=2):

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader


def train(model,
          device,
          train_loader,
          loss_func,
          optimizer,
          epoch,
          log_interval=10):

    model.train()
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        image_height = x.shape[2]
        image_width = x.shape[3]
        x = x.to(device)
        x = x.view(-1, image_height * image_width)
        optimizer.zero_grad()
        x_reconstructed, z, eps, log_std = model(x)
        loss = loss_func(x, x_reconstructed, z, eps, log_std)
        loss.backward()
        train_loss += loss.item() * len(x)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average Loss: {avg_train_loss:.4f}")


def test(model, device, num_samples, test_loader, loss_func, epoch,
         results_dir):

    image_dir = os.path.join(results_dir, "reconstruction")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(-1, model.num_observed_dims)
            x_reconstructed, z, eps, log_std = model(x)
            loss = loss_func(x, x_reconstructed, z, eps, log_std)
            test_loss += loss.item() * len(x)
            if i == 0:
                n = min(x.size(0), num_samples)
                comparison = torch.cat([
                    x.view(x.size(0), 1, 28, 28)[:n],
                    x_reconstructed.view(x.size(0), 1, 28, 28)[:n]
                ])
                torchvision.utils.save_image(
                    comparison.cpu(),
                    os.path.join(image_dir, f"reconstruction_{epoch}.png"),
                    nrow=n)
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"====> Test set loss: {avg_test_loss:.4f}")


def sample_random_images_using_std_normal_prior(model, device, num_samples,
                                                epoch, results_dir):

    image_dir = os.path.join(results_dir, "sample_using_std_normal_prior")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    model.eval()
    with torch.no_grad():
        sample = torch.randn(num_samples, model.num_latent_dims).to(device)
        sample = model.decoder(sample).cpu()
        torchvision.utils.save_image(
            sample.view(num_samples, 1, 28, 28),
            os.path.join(image_dir,
                         f"sample_using_std_normal_prior_{epoch}.png"))


def sample_random_images_using_2d_std_normal_prior_inverse_cdf(
        model, device, num_samples, epoch, results_dir):

    image_dir = os.path.join(results_dir,
                             "sample_using_2d_std_normal_prior_inverse_cdf")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    num_samples_per_dimension = int(math.sqrt(num_samples))
    cumulative_probability_samples = np.linspace(start=0.0001,
                                                 stop=0.9999,
                                                 num=num_samples_per_dimension)
    quantile_samples = torch.tensor([
        statistics.NormalDist(mu=0.0, sigma=1.0).inv_cdf(cp)
        for cp in cumulative_probability_samples
    ],
                                    dtype=torch.float32)

    model.eval()
    # Collect samples.
    samples = []
    with torch.no_grad():
        # Get z1 and z2.
        for i in range(num_samples_per_dimension):
            for j in range(num_samples_per_dimension):
                sample = torch.tensor(
                    [quantile_samples[i], quantile_samples[j]]).to(device)
                sample = sample.view(1, model.num_latent_dims)
                sample = model.decoder(sample).cpu()
                samples.append(sample)
    # Concatenate samples.
    samples = torch.cat(samples)
    # Save images. num_samples_per_dimension rows and columns.
    torchvision.utils.save_image(
        samples.view(num_samples, 1, 28, 28),
        os.path.join(
            image_dir,
            f"sample_using_2d_std_normal_prior_inverse_cdf_{epoch}.png"),
        nrow=num_samples_per_dimension)


def sample_random_images_using_reference_images(model, device, data_set,
                                                num_samples, epoch,
                                                results_dir):

    image_dir = os.path.join(results_dir, "sample_using_reference")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    reference_image_dir = os.path.join(results_dir, "reference")
    if not os.path.exists(reference_image_dir):
        os.makedirs(reference_image_dir)

    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(data_set), num_samples, replace=False)
        reference = torch.stack([data_set[i][0] for i in indices])
        reference = reference.to(device)
        reference = reference.view(-1, model.num_observed_dims)
        sample, _, _, _ = model(reference)
        torchvision.utils.save_image(
            sample.view(num_samples, 1, 28, 28),
            os.path.join(image_dir,
                         f"sample_using_reference_images_{epoch}.png"))
        torchvision.utils.save_image(
            reference.view(num_samples, 1, 28, 28),
            os.path.join(reference_image_dir, f"reference_images_{epoch}.png"))


def sample_ground_truth_images(data_set, num_samples, results_dir):

    indices = np.random.choice(len(data_set), num_samples, replace=False)
    sample = torch.stack([data_set[i][0] for i in indices])
    torchvision.utils.save_image(
        sample, os.path.join(results_dir, "ground_truth_sample.png"))


def main():

    cuda_device = torch.device("cuda:0")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    random_seed = 0
    set_random_seeds(random_seed=random_seed)

    mnist_image_height = 28
    mnist_image_width = 28

    num_observed_dims = mnist_image_height * mnist_image_width
    # This is a parameter to tune.
    # It should neither be too small nor too large.
    num_latent_dims = 2
    num_hidden_dims = 1024

    # 30 epochs is sufficient for MNIST and 2D manifold.
    num_epochs = 1
    learning_rate = 1e-3
    log_interval = 10

    train_set, test_set, class_names = prepare_cifar10_dataset(root=data_dir)

    sample_ground_truth_images(data_set=train_set,
                               num_samples=64,
                               results_dir=results_dir)

    train_loader, test_loader = prepare_cifar10_dataloader(
        train_set=train_set,
        test_set=test_set,
        train_batch_size=128,
        eval_batch_size=256,
        num_workers=2)

    model = VAE(num_observed_dims=num_observed_dims,
                num_latent_dims=num_latent_dims,
                num_hidden_dims=num_hidden_dims)
    model.to(cuda_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        train(model=model,
              device=cuda_device,
              train_loader=train_loader,
              loss_func=compute_negative_evidence_lower_bound,
              optimizer=optimizer,
              epoch=epoch,
              log_interval=log_interval)
        test(model=model,
             device=cuda_device,
             num_samples=16,
             test_loader=test_loader,
             loss_func=compute_negative_evidence_lower_bound,
             epoch=epoch,
             results_dir=results_dir)
        sample_random_images_using_std_normal_prior(model=model,
                                                    device=cuda_device,
                                                    num_samples=64,
                                                    epoch=epoch,
                                                    results_dir=results_dir)
        sample_random_images_using_reference_images(model=model,
                                                    device=cuda_device,
                                                    data_set=train_set,
                                                    num_samples=64,
                                                    epoch=epoch,
                                                    results_dir=results_dir)
        if num_latent_dims == 2:
            sample_random_images_using_2d_std_normal_prior_inverse_cdf(
                model=model,
                device=cuda_device,
                num_samples=400,
                epoch=epoch,
                results_dir=results_dir)

    # Save the model.
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    # Export the decoder to ONNX using Opset 13.
    z = torch.randn(1, num_latent_dims).to(cuda_device)
    torch.onnx.export(model.decoder,
                      z,
                      os.path.join(model_dir, "decoder.onnx"),
                      opset_version=13)


if __name__ == "__main__":

    main()
