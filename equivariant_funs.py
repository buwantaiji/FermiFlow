import torch
torch.set_default_dtype(torch.float64)

class Backflow(torch.nn.Module):
    """
        The backflow transformation that generates the collective coordinates
    {R_i} from the original ones {r_i}, where i = 1, ..., n, n being the total 
    particle number, and both R_i and r_i are dim-dimensional vectors, dim being 
    the space dimension. The transformation reads
        R_i = \sum_{j neq i} \eta(|r_i - r_j|) * (r_i - r_j).
    where \eta is any UNIVARIATE, SCALAR-VALUED function, possibly with some parameters. 

        It is easy to see that this transformation indeed serves as an equivariant 
    function respect to any permutation of particle positions.
    """
    def __init__(self, eta):
        """ The argument eta must be an instance of torch.nn.Module. """
        super(Backflow, self).__init__()
        self.eta = eta

    def forward(self, x):
        rij = x[:, :, None] - x[:, None]
        dij = rij.norm(dim=-1, keepdim=True)
        return (self.eta(dij) * rij).sum(dim=-2)

    def divergence(self, x):
        """
            The divergence is derived and coded by hand to avoid the computational
        overhead in CNF. The result is:
            div = \sum_{i neq j}^{n} ( \eta^prime(|r_i - r_j|) * |r_i - r_j|
                                        + dim * \eta(|r_i - r_j|) ).
        where \eta^prime denotes the derivative of the function \eta, n is the total
        particle number, and dim is the space dimension.
        """
        _, n, dim = x.shape
        row_indices, col_indices = torch.triu_indices(n, n, offset=1)

        rij = x[:, :, None] - x[:, None]
        dij = rij.norm(dim=-1, keepdim=True)[:, row_indices, col_indices, :]
        eta, d_eta = self.eta(dij), self.eta.grad(dij)
        div = 2 * (d_eta * dij + dim * eta).sum(dim=(-2, -1))
        return div

class FermiNet(torch.nn.Module):
    """
        The "backflow" part of the FermiNet architecture. For details, see
    https://arxiv.org/abs/1909.02487.
    """
    def __init__(self, n, dim, L, spsize, tpsize):
        """
            n: total number of particles.
            dim: space dimension.
            L: number of layers.
            spsize: the size of single-particle stream.
            tpsize: the size of two-particle stream.
        """
        super(FermiNet, self).__init__()
        self.n, self.dim = n, dim
        self.L = L
        spsize0 = tpsize0 = dim + 1

        self.spnet = torch.nn.ModuleList()
        self.tpnet = torch.nn.ModuleList()
        for i in range(self.L):
            real_spsize = spsize0 if i == 0 else spsize
            real_tpsize = tpsize0 if i == 0 else tpsize
            fsize = 2 * real_spsize + real_tpsize
            self.spnet.append(torch.nn.Linear(fsize, spsize))
            if (i != self.L-1):
                self.tpnet.append(torch.nn.Linear(real_tpsize, tpsize))

        self.final = torch.nn.Linear(spsize, dim)
        self.activation = torch.nn.Tanh()

    def _spstream0(self, x):
        """
            The initial single-particle stream: (r_i, |r_i|)
        shape: (batch, n, spsize0), where spsize0 = dim + 1.
        """
        return torch.cat( (x, x.norm(dim=-1, keepdim=True)), dim=-1 )

    def _tpstream0(self, x):
        """
            The initial two-particle stream: (r_ij, |r_ij|)
        shape: (batch, n, n, tpsize0), where tpsize0 = dim + 1.
        """
        rij = x[:, :, None] - x[:, None]
        return torch.cat( (rij, rij.norm(dim=-1, keepdim=True)), dim=-1 )

    def _f(self, spstream, tpstream):
        """
            The input to the sptream network, which is formed by concatenating the
        mean activations from both single and two particle streams in an equivariant way.
        shape: (batch, n, 2*spsize + tpsize)
        """
        return torch.cat( (spstream, 
                           spstream.mean(dim=-2, keepdim=True).expand(-1, self.n, -1), 
                           tpstream.mean(dim=-2)), dim=-1 )

    def forward(self, x):
        spstream, tpstream = self._spstream0(x), self._tpstream0(x)
        for i in range(self.L):
            f = self._f(spstream, tpstream)
            if (i == 0):
                spstream = self.activation(self.spnet[i](f))
                tpstream = self.activation(self.tpnet[i](tpstream))
            elif (i != self.L-1):
                spstream = self.activation(self.spnet[i](f)) + spstream
                tpstream = self.activation(self.tpnet[i](tpstream)) + tpstream
            else:
                spstream = self.activation(self.spnet[i](f)) + spstream
        output = self.final(spstream) + x
        return output
