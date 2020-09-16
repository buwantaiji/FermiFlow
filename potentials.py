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

class GaussianPairPotential(PairPotential):
    def __init__(self, g, s):
        super(GaussianPairPotential, self).__init__()
        self.g, self.s = g, s

    def V(self, x):
        rij = self.rij(x)
        return self.g / (np.pi * self.s**2) * torch.exp(- rij**2 / self.s**2).sum(dim=-1)

if __name__ == "__main__":
    g, s = 3.0, 0.5
    gaussianpotential = GaussianPairPotential(g, s)
    x = torch.randn(100, 7, 3)
    rij = gaussianpotential.rij(x)
    print(rij.shape)
    V = gaussianpotential.V(x)
    print(V.shape)

    sp_potential = HO()
    print(sp_potential.V(x).shape)
