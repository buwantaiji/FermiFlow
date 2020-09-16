import torch
torch.set_default_dtype(torch.float64)

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
