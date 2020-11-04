import torch
torch.set_default_dtype(torch.float64)

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

        _, x = self.sample((batch,))
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

class BetaVMC(torch.nn.Module):
    def __init__(self, beta, nup, ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=None):
        """
            Finite temperature Variational Monte Carlo calculation.
        """
        super(BetaVMC, self).__init__()

        self.beta = beta
        self.statistics = "Boson" if ndown is None else "Fermion"
        self.states = orbitals.fermion_states(nup, ndown) \
                        if self.statistics == "Fermion" else \
                        orbitals.boson_states(nup)
        self.Nstates = len(self.states)
        self.log_state_weights = torch.nn.Parameter(torch.randn(self.Nstates))

        self.basedist = basedist
        self.cnf = cnf

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential

    def sample(self, sample_shape):
        from torch.distributions.categorical import Categorical
        from collections import Counter
        if len(sample_shape) != 1:
            raise ValueError("BetaVMC.sample: sample_shape is required to have "
                    "only one batch dimension.")

        print("Sample state indices...")
        self.state_dist = Categorical(logits=self.log_state_weights)
        state_indices = self.state_dist.sample(sample_shape)
        self.state_indices_collection = Counter(sorted(state_indices.tolist()))
        zs = tuple( self.basedist.sample(*self.states[idx], (times,))
                for idx, times in self.state_indices_collection.items() )
        z = torch.cat(zs, dim=0)
        x = self.cnf.generate(z)
        return z, x 

    def logp(self, x, params_require_grad=False):
        z, delta_logp = self.cnf.delta_logp(x, params_require_grad=params_require_grad)
        log_prob_z = torch.empty_like(delta_logp)
        base_idx = 0
        for idx, times in self.state_indices_collection.items():
            log_prob_z[base_idx:base_idx+times] = \
                self.basedist.log_prob(*self.states[idx], z[base_idx:base_idx+times, ...])
            base_idx += times
        logp = log_prob_z - delta_logp
        return logp

    def forward(self, batch):
        from utils import y_grad_laplacian

        _, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = (kinetic + potential).detach()

        state_indices = torch.tensor(list(self.state_indices_collection.elements()), 
                            device=x.device)
        logp_states = self.state_dist.log_prob(state_indices)

        Floc = logp_states.detach() + self.beta * Eloc
        self.F, self.F_std = Floc.mean().item(), Floc.std().item()
        gradF_phi = (logp_states * (Floc - self.F)).mean()

        Eloc_x_mean = torch.empty_like(Eloc)
        base_idx = 0
        for idx, times in self.state_indices_collection.items():
            Eloc_x_mean[base_idx:base_idx+times] = Eloc[base_idx:base_idx+times].mean().expand(times)
            base_idx += times
        gradF_theta = self.beta * (logp_full * (Eloc - Eloc_x_mean)).mean()

        self.E, self.E_std = Eloc.mean().item(), Eloc.std().item()
        return gradF_phi, gradF_theta
