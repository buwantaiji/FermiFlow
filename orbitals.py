import torch
torch.set_default_dtype(torch.float64)
import numpy as np

class Orbitals(object):
    def __init__(self):
        pass
    def boson_states(self, n):
        pass
    def fermion_states(self, nup, ndown):
        """
            This function computes (a subset of) the low-lying eigenstates of the 
        non-interacting Fermion system, i.e., Slater determinants, under specific 
        single-particle hamiltonian.
            The arguments nup and ndown denote the number of spin-up and spin-down
        electrons, respectively. A tuple of several non-interacting states will be 
        returned, each represented by a 2-tuple of the form (orbitals_up, orbitals_down). 
        orbitals_up is itself a tuple of length nup, containing the nup occupied 
        single-particle orbitals, each of which is represented by a normal python 
        function. Similarly for orbitals_down.
            Note that due to Pauli exclusion principle, the nup(ndown) orbitals within
        orbitals_up(orbitals_down) are all different.
        """
        pass

class HO2D(Orbitals):
    """
        The single-particle orbitals of 2-dimensional isotropical harmonic potential.
    The hamiltonian reads: h(r) = - 1/2 laplacian + 1/2 r^2, where r = (x_1, x_2) is
    the coordinate vector in R^2.
    """
    def __init__(self):
        super(HO2D, self).__init__()

        pi_sqrt_inverse = 1. / np.sqrt(np.pi)
        orbitals_1d = [
            lambda x: 1, 
            lambda x: np.sqrt(2) * x, 
            lambda x: 1 / np.sqrt(2) * (2*x**2 - 1), 
            lambda x: 1 / np.sqrt(3) * (2*x**3 - 3*x), 
            lambda x: 1 / np.sqrt(6) * (2*x**4 - 6*x**2 + 1.5), 
            lambda x: 1 / np.sqrt(15) * (2*x**5 - 10*x**3 + 7.5*x), 
            lambda x: 1 / np.sqrt(5) * (2/3*x**6 - 5*x**4 + 7.5*x**2 - 1.25),
            ]
        orbital_2d = lambda nx, ny: lambda x: \
                        pi_sqrt_inverse * torch.exp(- 0.5 * (x**2).sum(dim=-1)) \
                        * orbitals_1d[nx](x[..., 0]) \
                        * orbitals_1d[ny](x[..., 1])

        self.orbitals = [orbital_2d(nx, n - nx) for n in range(7) for nx in range(n + 1)]
        self.Es = [n + 1 for n in range(7) for nx in range(n + 1)]
        self.E_indices = lambda n: tuple(range(n*(n+1)//2, (n+1)*(n+2)//2))

    def fermion_states_random(self, n):
        import random
        orbitals, Es = zip( *random.sample(tuple(zip(self.orbitals, self.Es)), k=n) )
        return orbitals, Es

    def fermion_states(self, nup, ndown, deltaE):
        import itertools

        if not (nup == 6 and ndown == 0):
            raise ValueError("nup and ndown must be 6 and 0 respectively "
                    "in the present implementation.")
        if deltaE > 4:
            raise ValueError("The maximum excitation energy deltaE of the states "
                    "is allowed to be at most 4 in the present implementation.")

        states = [(state, E) 
            for state, E in zip(itertools.combinations(self.orbitals, nup), 
                                itertools.combinations(self.Es, nup))
            if sum(E) <= sum(self.Es[:nup]) + deltaE]
        states = sorted(states, key=lambda state_E: sum(state_E[1]))
        states = tuple((state, ()) for state, _ in states)
        #states = tuple(sum(E) for _, E in states)
        return states

if __name__ == "__main__":
    ho2d = HO2D()
    for deltaE in range(5):
        states = ho2d.fermion_states(6, 0, deltaE)
        print(len(states))
