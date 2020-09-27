import torch
torch.set_default_dtype(torch.float64)

class BaseDist(object):
    """ The base class of base (i.e., prior) distribution. """
    def __init__(self, n, dim):
        """
            n: total particle numbers.
            dim: space dimension.
        """
        pass
    def sample(self, sample_shape):
        pass
    def log_prob(self, x):
        pass

class FreeBosonHO(BaseDist):
    """ 
        Ground state of n free Bosons trapped in an isotropical harmonic potential.
    The single particle hamiltonian reads: h(r) = - 1/2 laplacian + 1/2 r^2, 
    where r = (x_1, ..., x_d), d is the space dimension.

        The n-particle symmetric ground-state wavefunction reads: 
            \Psi_0(r_1, ..., r_n) = \prod_{i=1}^n \psi_0(r_i),
        where the single-particle ground-state wavefunction
            \psi_0(r) = 1 / pi^(d/4) * e^(-r^2/2).
    """

    def __init__(self, n, dim, device=torch.device("cpu")):
        from torch.distributions.normal import Normal
        from torch.distributions.independent import Independent

        super(FreeBosonHO, self).__init__(n, dim)

        self.dist = Normal(torch.zeros(n, dim, device=device), 
                            (0.5 * torch.ones(n, dim, device=device)).sqrt())
        self.dist = Independent(self.dist, reinterpreted_batch_ndims=2)

        self.sample= self.dist.sample
        self.log_prob = self.dist.log_prob

if __name__ == "__main__":
    import numpy as np 

    n, dim = 10, 3
    freebosonho = FreeBosonHO(n, dim)
    batch = 50
    x = freebosonho.sample((batch,))
    assert x.shape == (batch, n, dim)
    log_prob = freebosonho.log_prob(x)

    pi = torch.tensor(np.pi)

    log_prob_analytic = (- x**2 - 0.5 * torch.log(pi)).sum(dim=(-2, -1))
    assert torch.allclose(log_prob, log_prob_analytic)
