import torch
torch.set_default_dtype(torch.float64)

from flow import CNF

class GSVMC(torch.nn.Module):
    def __init__(self, nup, ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=None):
        """
            Ground State Variational Monte Carlo calculation.

        ---- INPUT ARGUMENTS ----

        nup, ndown: the number of spin-up and spin-down electrons in the Fermion
            case. If ndown is None, then the particles are understood to be spinless,
            which typically means that they are Bosons in this work.

        orbitals, basedist: orbitals contains the information of single-particle 
            orbitals and, combined with basedist, completely characterizes the base 
            distribution of the flow model.

        cnf: Continuous normalizing flow, which is an instance of the class CNF.
        """
        super(GSVMC, self).__init__()

        self.statistics = "Boson" if ndown is None else "Fermion"
        if self.statistics == "Fermion":
            self.orbitals_up, self.orbitals_down = orbitals.orbitals[:nup], \
                                                   orbitals.orbitals[:ndown]
        self.basedist = basedist
        self.cnf = cnf

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential

    def sample(self, sample_shape):
        z = self.basedist.sample(self.orbitals_up, self.orbitals_down, sample_shape) \
                if self.statistics == "Fermion" else \
                self.basedist.sample(sample_shape)
        x = self.cnf.generate(z)
        return z, x

    def logp(self, x, params_require_grad=False):
        z, delta_logp = self.cnf.delta_logp(x, params_require_grad=params_require_grad)
        logp = (self.basedist.log_prob(self.orbitals_up, self.orbitals_down, z) - delta_logp) \
                if self.statistics == "Fermion" else \
                (self.basedist.log_prob(z) - delta_logp)
        return logp

    def forward(self, batch):
        from utils import y_grad_laplacian

        z, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = (kinetic + potential).detach()

        self.E, self.E_std = Eloc.mean().item(), Eloc.std().item()
        gradE = (logp_full * (Eloc.detach() - self.E)).mean()
        return gradE
