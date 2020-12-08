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

####################################################################################
from slater import LogAbsSlaterDet, LogAbsSlaterDetMultStates

class FreeFermion(BaseDist):
    """ 
        This class serves to compute the log probability and sample the eigenstates
    of an non-interacting Fermion system, i.e., Slater determinants.

        For a non-interacting systems with nup spin-up electrons and ndown spin-down
    electrons, any eigenstate wavefunction (after eliminating the spin indices) 
    is written as the product of spin-up and spin-down Slater determinants:
        \Psi(r^up_1, ..., r^up_nup, r^down_1, r^down_ndown)
         = det(\phi^up_j(r^up_i)) * det(\phi^down_j(r^down_i)),
    where \phi^up_j (j = 1, ..., nup), \phi^down_j (j = 1, ..., ndown) are the occupied
    single-particle orbitals for the spin-up and spin-down electrons, respectively.
    These orbitals are passed as arguments "orbitals_up" and "orbitals_down" in the
    class methods.

        Note the wavefunction above is not normalized. The normalization factor is 
    1 / sqrt(nup! * ndown!).

    ================================================================================
    Below are a diagram demonstrating dependencies among the various functions.
    "1 --> 2" indicates function 2 depends on function 1. 

    LogAbsSlaterDet --> log_prob --> sample --> sample_multstates_old
                           |
                           v
             log_prob_multstates (method 1) --> sample_multstates (method 1)

    LogAbsSlaterDetMultStates --> log_prob_multstates (method 2) --> sample_multstates (method 2)
    """

    def __init__(self, device=torch.device("cpu")):
        super(FreeFermion, self).__init__()
        self.device = device

    def log_prob(self, orbitals_up, orbitals_down, x):
        nup, ndown = len(orbitals_up), len(orbitals_down)
        logabspsi = (LogAbsSlaterDet.apply(orbitals_up, x[..., :nup, :]) 
                        if nup != 0 else 0) \
                  + (LogAbsSlaterDet.apply(orbitals_down, x[..., nup:, :])
                        if ndown != 0 else 0)
        logp = 2 * logabspsi
        return logp

    def sample(self, orbitals_up, orbitals_down, sample_shape, 
            equilibrim_steps=100, tau=0.1):
        #print("Sample a Slater determinant...")
        nup, ndown = len(orbitals_up), len(orbitals_down)
        x = torch.randn(*sample_shape, nup + ndown, 2, device=self.device)
        logp = self.log_prob(orbitals_up, orbitals_down, x)
        for _ in range(equilibrim_steps):
            new_x = x + tau * torch.randn_like(x)
            new_logp = self.log_prob(orbitals_up, orbitals_down, new_x)
            p_accept = torch.exp(new_logp - logp)
            accept = torch.rand_like(p_accept) < p_accept
            x[accept] = new_x[accept]
            logp[accept] = new_logp[accept]
        return x

    def log_prob_multstates(self, states, state_indices_collection, x, method=2):
        if len(x.shape[:-2]) != 1:
            raise ValueError("FreeFermion.log_prob_multstates: x is required to have "
                    "only one batch dimension.")
        if method == 1:

            """ Making use of log_prob. """

            batch = x.shape[0]
            logp = torch.empty(batch, device=x.device)
            base_idx = 0
            for idx, times in state_indices_collection.items():
                logp[base_idx:base_idx+times] = \
                    self.log_prob(*states[idx], x[base_idx:base_idx+times, ...])
                base_idx += times
            return logp
        elif method == 2:

            """ Making use of the LogAbsSlaterDetMultStates primitive. """

            states_up, states_down = tuple(zip(*states))
            nup, ndown = len(states_up[0]), len(states_down[0])
            logabspsi = (LogAbsSlaterDetMultStates.apply(states_up, state_indices_collection, x[..., :nup, :]) 
                            if nup != 0 else 0) \
                      + (LogAbsSlaterDetMultStates.apply(states_down, state_indices_collection, x[..., nup:, :])
                            if ndown != 0 else 0)
            logp = 2 * logabspsi
            return logp

    def sample_multstates(self, states, state_indices_collection, sample_shape, 
            equilibrim_steps=100, tau=0.1, cpu=False, method=2):
        if len(sample_shape) != 1:
            raise ValueError("FreeFermion.sample_multstates: sample_shape is "
                    "required to have only one batch dimension.")

        #import time
        nup, ndown = len(states[0][0]), len(states[0][1])
        x = torch.randn(*sample_shape, nup + ndown, 2, 
                        device=torch.device("cpu") if cpu else self.device)
        logp = self.log_prob_multstates(states, state_indices_collection, x, method=method)
        #print("x.device:", x.device, "logp.device:", logp.device, "method:", method)

        for _ in range(equilibrim_steps):
            #start_out = time.time()

            new_x = x + tau * torch.randn_like(x)

            #start_in = time.time()
            new_logp = self.log_prob_multstates(states, state_indices_collection, new_x, method=method)
            #t_in = time.time() - start_in

            p_accept = torch.exp(new_logp - logp)
            accept = torch.rand_like(p_accept) < p_accept
            x[accept] = new_x[accept]
            logp[accept] = new_logp[accept]

            #t_out = time.time() - start_out
            #print("t_out:", t_out, "t_in:", t_in, "t_remain:", t_out - t_in, "ratio:", t_in / t_out)

        if cpu:
            x = x.to(device=self.device)
        return x

    def sample_multstates_old(self, states, state_indices_collection, sample_shape, 
            equilibrim_steps=100, tau=0.1):
        if len(sample_shape) != 1:
            raise ValueError("FreeFermion.sample_multstates_old: sample_shape is "
                    "required to have only one batch dimension.")

        xs = tuple( self.sample(*states[idx], (times,), 
                    equilibrim_steps=equilibrim_steps, tau=tau)
                for idx, times in state_indices_collection.items() )
        x = torch.cat(xs, dim=0)
        return x
