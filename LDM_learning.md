# Latent-diffusion Annotation
## Autoencoder

Calculation formula of kl divergence and negative log likelihood of normal distribution
```
class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
            
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
```
After processing the image into a hidden variable z, it is divided into two parts according to the channel dimension, one is mean and the other is logvar.

This code includes the calculation of KL Divergence, Negative log-likelihood

### KL Divergence Calculation

The KL divergence between two diagonal Gaussian distributions is calculated as follows:
  
  $$D_{KL}(N(\mu_1, \sigma_1^2) || N(\mu_2, \sigma_2^2)) = \frac{1}{2} \sum (\sigma_1^2 + (\mu_1 - \mu_2)^2) / \sigma_2^2 - 1 + \log(\sigma_2^2 / \sigma_1^2)$$

  where $\mu_1$, $\sigma_1^2$ are the mean and variance of the current distribution, and $\mu_2$, $\sigma_2^2$ are the mean and variance of the other distribution.

### Negative log-likelihood Calculation

The Negative log-likelihood is calculated as follows:

  $$NLL = \frac{n}{2} \log(2\pi) + \frac{n}{2} \log(\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2$$

  where $\mu$ is the sample mean, $\sigma$ is the variance of the current distribution.

## LDM
