import numpy as np
import torch
torch.set_default_dtype(torch.float64)

class SPPotential(object):
    def __init__(self):
        pass

class HO(SPPotential):
    def __init__(self):
        super(SPPotential, self).__init__()

    def V(self, x):
        return 0.5 * (x**2).sum(dim=(-2, -1))

# ==================================================================================

class PairPotential(object):
    
    def __init__(self):
        pass

    def rij(self, x):
        """
            Compute |r_i - r_j| for all 1 <= i < j <= n, n being the particle number.
            Input shape: (batch, n, dim)
            Output shape: (batch, n(n-1)/2)
        """
        n = x.shape[-2]
        row_indices, col_indices = torch.triu_indices(n, n, offset=1)
        return (x[:, :, None] - x[:, None])[:, row_indices, col_indices, :].norm(dim=-1)

    def V(self, x):
        """
            Compute the total potential energy V from the pair-wise potential function v:
            V = \sum_{i<j}^n v(|r_i - r_j|).
        """
        rij = self.rij(x)
        return self.v(rij).sum(dim=-1)

class CoulombPairPotential(PairPotential):
    def __init__(self, Z):
        super(CoulombPairPotential, self).__init__()
        self.Z = Z

    def v(self, rij):
        return self.Z / rij
