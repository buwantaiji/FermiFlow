import torch
torch.set_default_dtype(torch.float64)
import numpy as np

class Orbitals(object):
    def __init__(self):
        pass

    def fermion_states_random(self, n):
        import random
        orbitals, Es = zip( *random.sample(tuple(zip(self.orbitals, self.Es)), k=n) )
        return orbitals, Es

    def subsets(self, k, Pmax, Ps):
        """
            Given a set of several items with "prices" specified by the list Ps, find
        all subsets of length k whose total price do not exceed Pmax.
        """
        Nelements = len(Ps)
        result = ( ((), 0), )
        for i in range(1, k+1):
            result_new = []
            for subset, Ptotal in result:
                next_idx = subset[-1] + 1 if subset else 0
                while (next_idx + k - i < Nelements):
                    if sum(Ps[next_idx:next_idx+k-i+1]) <= Pmax - Ptotal:
                        result_new.append( (subset + (next_idx,), Ptotal + Ps[next_idx]) )
                    next_idx += 1
            result = tuple(result_new)
        indices, Ptotals = zip( *sorted(result, key=lambda index_P: index_P[1]) )
        return indices, Ptotals

    def fermion_states(self, nup, ndown, deltaE):
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
        if ndown != 0:
            raise ValueError("Only the polarized case (i.e., ndown = 0) is allowed "
                    "in the present implementation.")
        E0 = sum(self.Es[:nup])
        indices, Es = self.subsets(nup, E0 + deltaE, self.Es)
        states = tuple( (tuple(self.orbitals[idx] for idx in index_subset), ())
                        for index_subset in indices)
        return states, Es

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
            lambda x: 1 / np.sqrt(70) * (4/3*x**7 - 14*x**5 + 35*x**3 - 17.5*x),
            ]
        orbital_2d = lambda nx, ny: lambda x: \
                        pi_sqrt_inverse * torch.exp(- 0.5 * (x**2).sum(dim=-1)) \
                        * orbitals_1d[nx](x[..., 0]) \
                        * orbitals_1d[ny](x[..., 1])

        self.orbitals = [orbital_2d(nx, n - nx) for n in range(8) for nx in range(n + 1)]
        self.Es = [n + 1 for n in range(8) for nx in range(n + 1)]
        self.E_indices = lambda n: tuple(range(n*(n+1)//2, (n+1)*(n+2)//2))

    def fermion_states_naive(self, nup, ndown, deltaE):
        """
            A naive implementation of computing all the non-interacting many-body
        basis states of fermions, by exhaustive search of all possible combinations.
        """
        import itertools

        states = [(state, E) 
            for state, E in zip(itertools.combinations(self.orbitals, nup), 
                                itertools.combinations(self.Es, nup))
            if sum(E) <= sum(self.Es[:nup]) + deltaE]
        states, Es = zip( *sorted(states, key=lambda state_E: sum(state_E[1])) )
        states = tuple((state, ()) for state in states)
        Es = tuple(sum(E) for E in Es)
        return states, Es

if __name__ == "__main__":
    ho2d = HO2D()
    Ns = (3, 4, 6, 10)
    deltaEs_max = (2, 2, 4, 4)
    for N, deltaE_max in zip(Ns, deltaEs_max):
        print("---- N = %d ----" % N)
        for deltaE in range(deltaE_max + 1):
            states, Es = ho2d.fermion_states(N, 0, deltaE)
            print("deltaE =", deltaE, "Number of states:", len(states))
            print("State energies:", Es)
