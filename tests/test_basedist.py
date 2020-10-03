import torch
torch.set_default_dtype(torch.float64)

def test_FreeBosonHO():
    from base_dist import FreeBosonHO
    import numpy as np 

    n, dim = 10, 3
    freebosonho = FreeBosonHO(n, dim)
    batch = 50
    x = freebosonho.sample((batch,))
    assert x.shape == (batch, n, dim)
    log_prob = freebosonho.log_prob(x)

    pi = torch.tensor(np.pi)

    log_prob_analytic = (- x**2 - 0.5 * torch.log(pi)).sum(dim=(-2, -1))
    assert torch.allclose(log_prob, log_prob_analytic)

def test_slater_det():
    """ Test the antisymmetry property of the Slater determinant. """
    from base_dist import FreeFermionHO2D

    def slater_det(orbitals, x):
        *batch, n, _ = x.shape
        D = torch.empty(*batch, n, n)
        for i in range(n):
            D[..., i] = orbitals[i](x)
        return D.det()

    nup, ndown = 3, 6
    freefermionho2d = FreeFermionHO2D(nup, ndown)
    batch, n = (20, 30), nup + ndown
    x = torch.randn(*batch, n, 2)
    det_up = slater_det(freefermionho2d.orbitals_up, x[..., :nup, :])
    det_down = slater_det(freefermionho2d.orbitals_down, x[..., nup:, :])
    assert det_up.shape == batch and det_down.shape == batch

    P = torch.cat( (torch.randperm(nup), torch.randperm(ndown) + nup) )
    Px = x[..., P, :]

    Pdet_up = slater_det(freefermionho2d.orbitals_up, Px[..., :nup, :])
    Pdet_down = slater_det(freefermionho2d.orbitals_down, Px[..., nup:, :])

    assert torch.allclose(Pdet_up, det_up) or torch.allclose(Pdet_up, -det_up)
    assert torch.allclose(Pdet_down, det_down) or torch.allclose(Pdet_down, -det_down)

def test_slogdet():
    """ Test the backward of a hand-coded implementation of slogdet. """

    class MySlogdet(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            ctx.save_for_backward(A)
            _, logabsdet = A.slogdet() 
            return logabsdet
        @staticmethod
        def backward(ctx, grad_output):
            A, = ctx.saved_tensors
            return grad_output[..., None, None] * A.inverse().transpose(-2, -1)

    my_slogdet = MySlogdet.apply

    batch, n = (100, 70), 15
    x = torch.randn(*batch, n, n, requires_grad=True)

    _, logabsdet = x.slogdet()
    dlogabsdet, = torch.autograd.grad(logabsdet, x, grad_outputs=torch.ones(batch))
    assert logabsdet.shape == batch
    assert dlogabsdet.shape == (*batch, n, n)

    mylogabsdet = my_slogdet(x)
    dmylogabsdet, = torch.autograd.grad(mylogabsdet, x, grad_outputs=torch.ones(batch))

    assert torch.allclose(mylogabsdet, logabsdet)
    assert torch.allclose(dmylogabsdet, dlogabsdet)

def test_LogAbsSlaterDet():
    """ IMPORTANT TEST
        Test the result of backward of the LogAbsSlaterDet primitive is consistent
    with the approach logabsslaterdet, which works by directly backwarding through 
    the torch.slogdet function.
    """
    from base_dist import FreeFermionHO2D, LogAbsSlaterDet, logabsslaterdet
    log_abs_slaterdet = LogAbsSlaterDet.apply
    from utils import y_grad_laplacian

    nup, ndown = 3, 6
    freefermionho2d = FreeFermionHO2D(nup, ndown)
    batch = 20
    x = torch.randn(batch, nup, 2, requires_grad=True)

    logabsdet, grad_logabsdet, laplacian_logabsdet= \
        y_grad_laplacian(lambda x: log_abs_slaterdet(freefermionho2d.orbitals_up, x), x)
    assert logabsdet.shape == (batch,)
    assert grad_logabsdet.shape == (batch, nup, 2)
    assert laplacian_logabsdet.shape == (batch,)

    logabsdet_direct, grad_logabsdet_direct, laplacian_logabsdet_direct = \
        y_grad_laplacian(lambda x: logabsslaterdet(freefermionho2d.orbitals_up, x), x)
    assert torch.allclose(logabsdet_direct, logabsdet)
    assert torch.allclose(grad_logabsdet_direct, grad_logabsdet)
    assert torch.allclose(laplacian_logabsdet_direct, laplacian_logabsdet)

def test_FreeFermionHO2D_slaterdet():
    """ IMPORTANT TEST
        Test the single-particle orbitals and Slater determinants constructed
    in FreeFermionHO2D base distribution class are indeed eigenfunctions of the
    corresponding Hamiltonians.
    """
    from base_dist import FreeFermionHO2D
    from utils import y_grad_laplacian

    nup, ndown = 3, 6
    freefermionho2d = FreeFermionHO2D(nup, ndown)
    batch, n = 20, nup + ndown

    """ Test the single-particle orbitals. """
    i = 0
    x = torch.randn(batch, 2, requires_grad=True)
    log, grad_log, laplacian_log = y_grad_laplacian(
        lambda x: freefermionho2d.OrbitalsHO2D[i](x).log(), x)
    kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=-1)
    potential = 0.5 * (x**2).sum(dim=-1)
    Eloc = kinetic + potential
    assert torch.allclose(Eloc, freefermionho2d.Es[i] * torch.ones(batch))

    """ Test a single Slater determinant. """
    from base_dist import LogAbsSlaterDet
    log_abs_slaterdet = LogAbsSlaterDet.apply

    x = torch.randn(batch, nup, 2, requires_grad=True)
    log, grad_log, laplacian_log = y_grad_laplacian(
        lambda x: log_abs_slaterdet(freefermionho2d.OrbitalsHO2D[:nup], x), x)
    kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    assert torch.allclose(Eloc, sum(freefermionho2d.Es[:nup]) * torch.ones(batch))

    """ Test a complete Fermion wavefunction composed of two Slater determinants. """
    x = torch.randn(batch, n, 2, requires_grad=True)
    logp, grad_logp, laplacian_logp = y_grad_laplacian(freefermionho2d.log_prob, x)
    kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    assert torch.allclose(Eloc, sum(freefermionho2d.Es[:nup]
                                  + freefermionho2d.Es[:ndown]) * torch.ones(batch))

def test_FreeFermionHO2D_sample():
    from base_dist import FreeFermionHO2D

    nup, ndown = 3, 6
    freefermionho2d = FreeFermionHO2D(nup, ndown)
    batch, n = (20, 30), nup + ndown
    x = torch.randn(*batch, n, 2)

    logp = freefermionho2d.log_prob(x)
    assert logp.shape == batch

    samples = freefermionho2d.sample(batch)
    assert samples.shape == (*batch, n, 2)
