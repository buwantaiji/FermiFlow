import torch
torch.set_default_dtype(torch.float64)

class Backflow(torch.nn.Module):
    """
        The backflow transformation that generates the collective coordinates
    {xi_i} from the original ones {r_i}, where i = 1, ..., n, n being the total 
    particle number, and both xi_i and r_i are dim-dimensional vectors, dim being 
    the space dimension.
    """
    def __init__(self, eta, mu=None):
        """ The argument eta must be an instance of torch.nn.Module. """
        super(Backflow, self).__init__()
        self.eta = eta
        self.mu = mu

    def _e_e(self, x):
        """
            The two-body part xi^{e-e}_i of the backflow transformation, which
        takes cares of the two-body correlation of the system. It reads as follows:
            xi^{e-e}_i = \\sum_{j neq i} eta(|r_i - r_j|) * (r_i - r_j).
        where eta is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 
        """
        _, n, dim = x.shape

        rij = x[:, :, None] - x[:, None]
        rij += torch.eye(n, device=x.device)[:, :, None]
        dij = rij.norm(dim=-1, keepdim=True)
        output = (self.eta(dij) * rij).sum(dim=-2)
        output -= self.eta(torch.ones(dim, device=x.device).norm()[None])
        return output

    def _e_e_divergence(self, x):
        """
            The divergence of the two-body part xi^{e-e}_i of the transformation, 
        which is derived and coded by hand to avoid the computational overhead in CNF. 
        The result is:
            div^{e-e} = \\sum_{i neq j}^{n} ( eta^prime(|r_i - r_j|) * |r_i - r_j|
                                        + dim * eta(|r_i - r_j|) ).
        where eta^prime denotes the derivative of the function eta, n is the total
        particle number, and dim is the space dimension.
        """
        _, n, dim = x.shape
        row_indices, col_indices = torch.triu_indices(n, n, offset=1)

        rij = x[:, :, None] - x[:, None]
        dij = rij.norm(dim=-1, keepdim=True)[:, row_indices, col_indices, :]
        eta, d_eta = self.eta(dij), self.eta.grad(dij)
        div_e_e = 2 * (d_eta * dij + dim * eta).sum(dim=(-2, -1))
        return div_e_e

    def _e_n(self, x):
        """
            The one-body (i.e., mean-field) part xi^{e-n}_i of the backflow 
        transformation, which takes cares of the interaction of one particle with
        some "nuclei" positions in the system, possibly arising from the nuclei in 
        a real molecule or harmonic trap in cold-atom systems, and so on.
            For simplicity, it is assumed that there is only one nucleus position
        at the origin. Then the transformation reads as follows:
            xi^{e-n}_i = mu(|r_i|) * r_i.
        where mu is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 
        """
        di = x.norm(dim=-1, keepdim=True)
        return self.mu(di) * x

    def _e_n_divergence(self, x):
        """
            The divergence of the one-body part xi^{e-n}_i of the transformation, 
        which is derived and coded by hand to avoid the computational overhead in CNF. 
        The result (for the simplified single-nucleus case) is:
            div^{e-n} = \\sum_{i=1}^{n} ( mu^prime(|r_i|) * |r_i| 
                                        + dim * mu(|r_i|) ).
        where mu^prime denotes the derivative of the function mu, n is the total
        particle number, and dim is the space dimension.
        """
        dim = x.shape[-1]

        di = x.norm(dim=-1, keepdim=True)
        mu, d_mu = self.mu(di), self.mu.grad(di)
        div_e_n = ( d_mu * di + dim * mu ).sum(dim=(-2, -1))
        return div_e_n

    def forward(self, x):
        """
            The total backflow transformation xi_i, which contains the two-body part
        and (possibly) the one-body part:
            xi_i = xi^{e-e}_i + xi^{e-n}_i.

            It is easy to see that both components serve as equivariant functions 
        respect to any permutation of particle positions, then so do their sum.
        """
        return self._e_e(x) + \
              (self._e_n(x) if self.mu is not None else 0)

    def divergence(self, x):
        """
            The divergence of the total backflow transformation, which contains the 
        two-body part and (possibly) the one-body part:
            div = div^{e-e} + div^{e-n}.
        """
        return self._e_e_divergence(x) + \
              (self._e_n_divergence(x) if self.mu is not None else 0)
