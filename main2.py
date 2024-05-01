import math
import torch
import torch.nn as nn
import torchvision

from torch.distributions.multivariate_normal import MultivariateNormal

class VariationalEncoder(nn.Module):
    
    def __init__(self, num_observed_dims=784, num_latent_dims=8, num_hidden_dims=512):
        super(VariationalEncoder, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.fc = nn.Linear(in_features=self.num_observed_dims, out_features=self.num_hidden_dims)
        # Encoded variational mean.
        self.fc_mean = nn.Linear(in_features=self.num_hidden_dims, out_features=self.num_latent_dims)
        # Encoded variational log standard deviation.
        self.fc_log_std = nn.Linear(in_features=self.num_hidden_dims, out_features=self.num_latent_dims)
        # Encoded flattened unmasked lower triangular matrix.
        self.fc_unmasked_lower_triangular_flatten = nn.Linear(in_features=self.num_hidden_dims, out_features=self.num_latent_dims * self.num_latent_dims)
        # Constant mask for lower triangular matrix.
        self.mask = torch.tril(torch.ones(self.num_latent_dims, self.num_latent_dims), diagonal=-1)

        self.multivariate_std_normal = MultivariateNormal(torch.zeros(self.num_latent_dims), torch.eye(self.num_latent_dims))
    
    def encode(self, x):
            
        h = torch.relu(self.fc(x))
        mu = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        unmasked_lower_triangular_flatten = self.fc_unmasked_lower_triangular_flatten(h)
        unmasked_lower_triangular = unmasked_lower_triangular_flatten.view(-1, self.num_latent_dims, self.num_latent_dims)

        return mu, log_std, unmasked_lower_triangular

    def reparameterize(self, mu, log_std, unmasked_lower_triangular):

        # Using full-covariance Gaussian posterior
        # TODO: This only works for BS = 1, need to implement for BS >= 1.
        std = torch.exp(log_std)
        # torch.diag_embed diagonalizes the vector in batches.
        lower_triangular = unmasked_lower_triangular * self.mask + torch.diag_embed(std)
        # print(lower_triangular.shape)
        # Sample from standard normal in batch.
        eps = self.multivariate_std_normal.sample(sample_shape=torch.Size([mu.shape[0]]))
        # print(eps.shape)
        # print(torch.bmm(lower_triangular, eps).shape)
        z = mu + torch.bmm(lower_triangular, eps.view(-1, self.num_latent_dims, 1)).view(-1, self.num_latent_dims)

        return z, eps
    
    def forward(self, x):

        mu, log_std, unmasked_lower_triangular = self.encode(x)
        z, eps = self.reparameterize(mu, log_std, unmasked_lower_triangular)

        return z, eps, log_std

class Decoder(nn.Module):
    
    def __init__(self, num_observed_dims=784, num_latent_dims=8, num_hidden_dims=512):
        super(Decoder, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.fc = nn.Linear(in_features=self.num_latent_dims, out_features=self.num_hidden_dims)
        self.fc_out = nn.Linear(in_features=self.num_hidden_dims, out_features=self.num_observed_dims)

    def decode(self, z):

        h = torch.relu(self.fc(z))
        x = torch.sigmoid(self.fc_out(h))

        return x
    
    def forward(self, z):

        return self.decode(z)


class VAE(nn.Module):

    def __init__(self, num_observed_dims=784, num_latent_dims=8, num_hidden_dims=512):
        super(VAE, self).__init__()

        self.num_observed_dims = num_observed_dims
        self.num_latent_dims = num_latent_dims
        self.num_hidden_dims = num_hidden_dims

        self.encoder = VariationalEncoder(num_observed_dims=self.num_observed_dims, num_latent_dims=self.num_latent_dims, num_hidden_dims=self.num_hidden_dims)
        self.decoder = Decoder(num_observed_dims=self.num_observed_dims, num_latent_dims=self.num_latent_dims, num_hidden_dims=self.num_hidden_dims)

    def forward(self, x):
            
        z, eps, log_std = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, eps, log_std


def compute_negative_evidence_lower_bound(x, x_reconstructed, z, eps, log_std):

    # Reconstruction loss.
    # E[log p(x|z)]
    log_px = - torch.nn.functional.binary_cross_entropy(x_reconstructed, x, reduction="sum")
    # E[log q(z|x)]
    log_qz = - 0.5 * torch.sum(eps ** 2 + log_std + torch.log(2 * math.pi))
    # E[log p(z)]
    log_pz = - 0.5 * torch.sum(z ** 2 + torch.log(2 * math.pi))
    # ELBO
    elbo = log_px + log_pz - log_qz
    negative_elbo = - elbo

    # Compute average negative ELBO.
    batch_size = x.shape[0]
    negative_elbo_avg = negative_elbo / batch_size

    return negative_elbo_avg


def prepare_cifar10_dataloader(num_workers=2,
                               train_batch_size=128,
                               eval_batch_size=256):

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)

    test_set = torchvision.datasets.MNIST(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

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

    class_names = train_set.classes

    return train_loader, test_loader, class_names

def main():

    train_loader, test_loader, class_names = prepare_cifar10_dataloader(num_workers=1, train_batch_size=128, eval_batch_size=256)

    vae = VAE()
    for data in train_loader:
        images, labels = data
        images = images.view(-1, 784)
        x_reconstructed, z, eps, log_std = vae(images)
        loss = compute_negative_evidence_lower_bound(images, x_reconstructed, z, eps, log_std)
        print(loss)
        



if __name__ == "__main__":
    
    main()