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
    def __init__(self, beta, nup, ndown, deltaE, orbitals, basedist, cnf, 
                 pair_potential, sp_potential=None):
        """
            Finite temperature Variational Monte Carlo calculation.

        ---- NOTABLE ARGUMENTS ----
            
        deltaE: The maximum excitation energy of the truncated states. In the present
            implementation, the case of Fermions trapped in 2D harmonic potential is
            considered, and deltaE takes value up to 3. See orbitals.py for details.
        """
        super(BetaVMC, self).__init__()

        self.beta = beta
        if ndown is None:
            self.statistics = "Boson"
            self.states, self.Es_original = orbitals.boson_states(nup, deltaE)
        else:
            self.statistics = "Fermion"
            self.states, self.Es_original = orbitals.fermion_states(nup, ndown, deltaE)
        self.Es_original = torch.tensor(self.Es_original)
        self.Nstates = len(self.states)
        self.log_state_weights = torch.nn.Parameter(torch.randn(self.Nstates))

        self.basedist = basedist
        self.cnf = cnf

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential

    def sample(self, sample_shape):
        from torch.distributions.categorical import Categorical
        from collections import Counter
        import time

        self.state_dist = Categorical(logits=self.log_state_weights)
        state_indices = self.state_dist.sample(sample_shape)
        self.state_indices_collection = Counter(sorted(state_indices.tolist()))

        start = time.time()
        z = self.basedist.sample_multstates(self.states, 
                self.state_indices_collection, sample_shape)
        print("Finished sampling basis states. Time to take (hours per 100 iters):", 
                (time.time() - start) * 100 / 3600)

        x = self.cnf.generate(z)
        return z, x 

    def logp(self, x, params_require_grad=False):
        z, delta_logp = self.cnf.delta_logp(x, params_require_grad=params_require_grad)

        log_prob_z = self.basedist.log_prob_multstates(self.states, 
                self.state_indices_collection, z)

        logp = log_prob_z - delta_logp
        return logp

    def compute_energies(self, sample_shape, device):
        if self.statistics != "Fermion":
            raise ValueError("BetaVMC.compute_energies: only fermion statistics is "
                    "allowed in the present implementation.")

        from utils import y_grad_laplacian
        def logp_singlestate(x, orbitals_up, orbitals_down):
            z, delta_logp = self.cnf.delta_logp(x, params_require_grad=False)
            logp = self.basedist.log_prob(orbitals_up, orbitals_down, z) - delta_logp
            return logp

        Es_flow = torch.empty(self.Nstates, device=device)
        Es_std_flow = torch.empty(self.Nstates, device=device)
        for idx, (orbitals_up, orbitals_down) in enumerate(self.states):
            z = self.basedist.sample(orbitals_up, orbitals_down, sample_shape)
            x = self.cnf.generate(z)
            x.requires_grad_(True)

            logp, grad_logp, laplacian_logp = y_grad_laplacian(
                    lambda x: logp_singlestate(x, orbitals_up, orbitals_down), x)
            kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

            potential = self.pair_potential.V(x)
            if self.sp_potential:
                potential += self.sp_potential.V(x)

            Eloc = (kinetic + potential).detach()
            Es_flow[idx], Es_std_flow[idx] = Eloc.mean(), Eloc.std()
            print(idx, self.Es_original[idx].item(), Es_flow[idx].item())
        return Es_flow, Es_std_flow

    def forward(self, batch):
        """
            Physical quantities of interest:
        self.E, self.E_std: mean and standard deviation of energy.
        self.F, self.F_std: mean and standard deviation of free energy.
        self.S, self.S_analytical: entropy of the system, computed using Monte Carlo
            sampling and the direct formula of von-Neumann, respectively.
        self.logp_states_all: lop-probability of each of the considered states, 
            which is represented by a 1D tensor of size self.Nstates.
        """
        from utils import y_grad_laplacian
        import time

        _, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        start = time.time()
        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        print("Computed gradients of logp up to 2nd order. "
                "Time to take (hours per 100 iters):", 
                (time.time() - start) * 100 / 3600)

        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = (kinetic + potential).detach()
        self.E, self.E_std = Eloc.mean().item(), Eloc.std().item()

        state_indices = torch.tensor(list(self.state_indices_collection.elements()), 
                            device=x.device)
        logp_states = self.state_dist.log_prob(state_indices)

        Floc = Eloc + logp_states.detach() / self.beta
        self.F, self.F_std = Floc.mean().item(), Floc.std().item()

        self.S = -logp_states.detach().mean().item()
        self.logp_states_all = self.state_dist.log_prob(torch.arange(self.Nstates, 
                            device=x.device)).detach()
        self.S_analytical = -(self.logp_states_all * 
                              self.logp_states_all.exp()).sum().item()

        gradF_phi = (logp_states * (Floc - self.F)).mean()

        Eloc_x_mean = torch.empty_like(Eloc)
        base_idx = 0
        for idx, times in self.state_indices_collection.items():
            Eloc_x_mean[base_idx:base_idx+times] = Eloc[base_idx:base_idx+times].mean().expand(times)
            base_idx += times
        gradF_theta = (logp_full * (Eloc - Eloc_x_mean)).mean()

        return gradF_phi, gradF_theta
