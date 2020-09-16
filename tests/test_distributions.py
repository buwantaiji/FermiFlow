import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent

pi = torch.tensor(np.pi)

def normal_logp(x, mean, std):
    """
        Note that in the most general case, x has shape (*sample_shape, *batch_shape), 
    whereas both mean and std have shape batch_shape. Broadcast rules will generally
    apply here!
    """
    return - (x - mean)**2 / (2 * std**2) \
            - torch.log(torch.sqrt(2 * pi) * std)

def test_normal():
    batch_shape = (2, 3, 5)
    means = 10 * torch.arange(np.prod(batch_shape)).reshape(batch_shape)
    stds = torch.rand(batch_shape)

    dist = Normal(means, stds)
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == ()

    x = dist.sample()
    assert x.shape == dist.batch_shape
    logp = dist.log_prob(x)
    assert logp.shape == x.shape
    assert torch.allclose(logp, normal_logp(x, means, stds))

    sample_shape = (14,)
    x = dist.sample(sample_shape)
    assert x.shape == sample_shape + dist.batch_shape
    logp = dist.log_prob(x)
    assert logp.shape == x.shape
    assert torch.allclose(logp, normal_logp(x, means, stds))

def multivariate_normal_logp(x, mean, diag):
    """
        In the most general case, x has shape (*sample_shape, *batch_shape, *event_shape), 
    whereas both mean and diag have shape (*batch_shape, *event_shape). Broadcast rules
    will generally apply here!
        Also note that due to the nature of multivariate normal distribution, the
    event_shape has only one dimension.
    """
    d = diag.shape[-1]
    return -0.5 * ((x - mean) / diag * (x - mean)).sum(dim=-1) \
            - 0.5 * d * torch.log(2 * pi) - 0.5 * torch.log(diag).sum(dim=-1)

def test_multivariate_normal():
    batch_shape = (7, 9)
    d = 2
    means = 10 * torch.randn(*batch_shape, d)
    diag = 0.5 + torch.rand(d)
    covariances = torch.diag(diag)

    dist = MultivariateNormal(means, covariances)
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == (d,)

    x = dist.sample()
    assert x.shape == dist.batch_shape + dist.event_shape
    logp = dist.log_prob(x)
    assert logp.shape == dist.batch_shape
    assert torch.allclose(logp, multivariate_normal_logp(x, means, diag))

    sample_shape = (40, 30)
    x = dist.sample(sample_shape)
    assert x.shape == sample_shape + dist.batch_shape + dist.event_shape
    logp = dist.log_prob(x)
    assert logp.shape == sample_shape + dist.batch_shape
    assert torch.allclose(logp, multivariate_normal_logp(x, means, diag))

def test_independent():
    batch_shape = (8, 10, 3)
    means = 10 * torch.randn(batch_shape)
    stds = 0.5 + torch.rand(batch_shape)
    dist = Normal(means, stds)
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == ()

    dist_independent = Independent(dist, reinterpreted_batch_ndims=1)
    assert dist_independent.batch_shape == batch_shape[:-1]
    assert dist_independent.event_shape == batch_shape[-1:]
    sample_shape = (15, 25)
    x = dist_independent.sample(sample_shape)
    assert x.shape == sample_shape + batch_shape

    dist_equivalent = MultivariateNormal(means, scale_tril=torch.diag_embed(stds))
    assert dist_independent.batch_shape == dist_equivalent.batch_shape
    assert dist_independent.event_shape == dist_equivalent.event_shape
    logp1 = dist_independent.log_prob(x)
    logp2 = dist_equivalent.log_prob(x)
    assert logp1.shape == sample_shape + dist_independent.batch_shape
    assert torch.allclose(logp1, logp2)
