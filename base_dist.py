import torch
torch.set_default_dtype(torch.float64)

class BaseDist(object):
    """ The base class of base (i.e., prior) distribution. """
    def __init__(self):
        pass
    def log_prob(self, x):
        pass
    def sample(self, sample_shape):
        pass

class FreeBosonHO(BaseDist):
    """ 
        Ground state of n free Bosons trapped in an isotropical harmonic potential.
    The single-particle hamiltonian reads: h(r) = - 1/2 laplacian + 1/2 r^2, 
    where r = (x_1, ..., x_d), d is the space dimension.

        The n-particle symmetric ground-state wavefunction reads: 
            \Psi_0(r_1, ..., r_n) = \prod_{i=1}^n \psi_0(r_i),
        where the single-particle ground-state wavefunction
            \psi_0(r) = 1 / pi^(d/4) * e^(-r^2/2).
    """

    def __init__(self, n, dim, device=torch.device("cpu")):
        """
            n: total particle numbers.
            dim: space dimension.
        """
        from torch.distributions.normal import Normal
        from torch.distributions.independent import Independent

        super(FreeBosonHO, self).__init__()

        self.dist = Normal(torch.zeros(n, dim, device=device), 
                            (0.5 * torch.ones(n, dim, device=device)).sqrt())
        self.dist = Independent(self.dist, reinterpreted_batch_ndims=2)

        self.log_prob = self.dist.log_prob
        self.sample= self.dist.sample

class FreeFermionHO2D(BaseDist):
    """ 
        Ground state of n free 2D Fermions trapped in an isotropical harmonic potential.
    The single-particle hamiltonian reads: h(r) = - 1/2 laplacian + 1/2 r^2, 
    where r = (x_1, x_2) is the coordinate vector in R^2.

        Users should specify nup, ndown, the number of spin up and spin down electrons, 
    respectively. Then the ground state wavefunction (after eliminating the spin indices)
    is represented by the product of spin-up and spin-down Slater determinants:
        \Psi_0(r^up_1, ..., r^up_nup, r^down_1, r^down_ndown)
         = det(\phi^up_j(r^up_i)) * det(\phi^down_j(r^down_i)),
    where \phi^up_j (j = 1, ..., nup), \phi^down_j (j = 1, ..., ndown) are the
    single-particle orbitals for the spin-up and spin-down electrons, respectively.

    The expression above is not normalized. The normalization factor is 1 / sqrt(nup! * ndown!).
    """

    def __init__(self, nup, ndown):
        import numpy as np

        super(FreeFermionHO2D, self).__init__()
        pi1_over_4_inverse = 1. / np.pi ** (1/4)
        OrbitalsHO1D = [
            lambda x: pi1_over_4_inverse, 
            lambda x: pi1_over_4_inverse * np.sqrt(2) * x, 
            lambda x: pi1_over_4_inverse / np.sqrt(2) * (2*x**2 - 1), 
            lambda x: pi1_over_4_inverse / np.sqrt(3) * (2*x**3 - 3*x)]
        HO2D = lambda x, nx, ny: torch.exp(- 0.5 * (x**2).sum(dim=-1)) \
                                * OrbitalsHO1D[nx](x[..., 0]) \
                                * OrbitalsHO1D[ny](x[..., 1])
        OrbitalsHO2D = [
            lambda x: HO2D(x, 0, 0), 
            lambda x: HO2D(x, 0, 1),
            lambda x: HO2D(x, 1, 0),
            lambda x: HO2D(x, 0, 2), 
            lambda x: HO2D(x, 1, 1), 
            lambda x: HO2D(x, 2, 0)]

        self.nup, self.ndown = nup, ndown
        self.orbitals_up, self.orbitals_down = OrbitalsHO2D[:nup], OrbitalsHO2D[:ndown]

    def log_prob(self, x):
        logabspsi = (LogAbsSlaterDet.apply(self.orbitals_up, x[..., :self.nup, :]) 
                        if self.nup != 0 else 0) \
                  + (LogAbsSlaterDet.apply(self.orbitals_down, x[..., self.nup:, :])
                        if self.ndown != 0 else 0)
        logp = 2 * logabspsi
        return logp

    def sample(self, sample_shape, equilibrim_steps=100, tau=0.05):
        x = torch.randn(*sample_shape, self.nup + self.ndown, 2)
        logp = self.log_prob(x)
        for _ in range(equilibrim_steps):
            new_x = x + tau * torch.randn_like(x)
            new_logp = self.log_prob(new_x)
            p_accept = torch.exp(new_logp - logp)
            accept = torch.rand_like(p_accept) < p_accept
            x[accept] = new_x[accept]
            logp[accept] = new_logp[accept]
        return x

class LogAbsSlaterDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orbitals, x):
        """
            --- INPUT ---

            orbitals: a tuple of length n containing the n orbitals {\phi_i(r)}, 
            each of which is represented by a normal function.
            
            x: the particle coordinates. Generally, x has shape (*batch, n, dim),
               where several batch dimensions are allowed.

            --- OUTPUT ---

            The nxn slater determinant det(\phi_j(r_i)), which has shape (*batch)
            in the general case.
        """
        *batch, n, _ = x.shape
        D = torch.empty(*batch, n, n)
        for i in range(n):
            D[..., i] = orbitals[i](x)

        ctx.save_for_backward(D, x)
        ctx.orbitals = orbitals

        _, logabsdet = D.slogdet()
        return logabsdet
    
    @staticmethod
    def backward(ctx, grad_logabsdet):
        D, x = ctx.saved_tensors
        orbitals = ctx.orbitals
        *batch, n, dim = x.shape

        dorbitals = torch.empty(*batch, n, dim, n)
        with torch.enable_grad():
            for i in range(n):
                orbital_value = orbitals[i](x)
                dorbitals[..., i], = torch.autograd.grad(orbital_value, x, 
                            grad_outputs=torch.ones_like(orbital_value))
        dlogabsdet = torch.einsum("...ndj,...jn->...nd", dorbitals, D.inverse())
        grad_x = grad_logabsdet[..., None, None] * dlogabsdet
        return None, grad_x
