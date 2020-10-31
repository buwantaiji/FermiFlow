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
    from orbitals import HO2D
    ho2d = HO2D()

    def slater_det(orbitals, x):
        *batch, n, _ = x.shape
        D = torch.empty(*batch, n, n)
        for i in range(n):
            D[..., i] = orbitals[i](x)
        return D.det()

    nup, ndown = 3, 6
    orbitals_up, _ = ho2d.fermion_states_random(nup)
    orbitals_down, _ = ho2d.fermion_states_random(ndown)

    batch, n = (20, 30), nup + ndown
    x = torch.randn(*batch, n, 2)
    det_up = slater_det(orbitals_up, x[..., :nup, :])
    det_down = slater_det(orbitals_down, x[..., nup:, :])
    assert det_up.shape == batch and det_down.shape == batch

    P = torch.cat( (torch.randperm(nup), torch.randperm(ndown) + nup) )
    Px = x[..., P, :]

    Pdet_up = slater_det(orbitals_up, Px[..., :nup, :])
    Pdet_down = slater_det(orbitals_down, Px[..., nup:, :])

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
    from orbitals import HO2D
    from base_dist import LogAbsSlaterDet, logabsslaterdet
    log_abs_slaterdet = LogAbsSlaterDet.apply
    from utils import y_grad_laplacian

    ho2d = HO2D()
    n = 3
    orbitals, _ = ho2d.fermion_states_random(n)
    batch = 20
    x = torch.randn(batch, n, 2, requires_grad=True)

    logabsdet, grad_logabsdet, laplacian_logabsdet= \
        y_grad_laplacian(lambda x: log_abs_slaterdet(orbitals, x), x)
    assert logabsdet.shape == (batch,)
    assert grad_logabsdet.shape == (batch, n, 2)
    assert laplacian_logabsdet.shape == (batch,)

    logabsdet_direct, grad_logabsdet_direct, laplacian_logabsdet_direct = \
        y_grad_laplacian(lambda x: logabsslaterdet(orbitals, x), x)
    assert torch.allclose(logabsdet_direct, logabsdet)
    assert torch.allclose(grad_logabsdet_direct, grad_logabsdet)
    assert torch.allclose(laplacian_logabsdet_direct, laplacian_logabsdet)

def test_FreeFermionHO2D_slaterdet():
    """ IMPORTANT TEST
        Test the single-particle orbitals constructed in HO2D and the corresponding 
    many-body Slater determinants are indeed eigenfunctions of the 2D harmonic
    oscillator Hamiltonian.
    """
    from orbitals import HO2D
    from utils import y_grad_laplacian

    ho2d = HO2D()
    Norbitals = len(ho2d.orbitals)
    batch = 20

    """ Test the single-particle orbitals. """
    for orbital, E in zip(ho2d.orbitals, ho2d.Es):
        x = torch.randn(batch, 2, requires_grad=True)
        log, grad_log, laplacian_log = y_grad_laplacian(
            lambda x: orbital(x).log(), x)
        kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=-1)
        potential = 0.5 * (x**2).sum(dim=-1)
        Eloc = kinetic + potential
        assert torch.allclose(Eloc, E * torch.ones(batch))

    """ Test a single Slater determinant. """
    from base_dist import LogAbsSlaterDet
    log_abs_slaterdet = LogAbsSlaterDet.apply

    n = 5
    orbitals, Es = ho2d.fermion_states_random(n)
    x = torch.randn(batch, n, 2, requires_grad=True)
    log, grad_log, laplacian_log = y_grad_laplacian(
        lambda x: log_abs_slaterdet(orbitals, x), x)
    kinetic = - 1/2 * laplacian_log - 1/2 * (grad_log**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    #print("Eloc:", Eloc, "\nEs:", Es)
    assert torch.allclose(Eloc, sum(Es) * torch.ones(batch))

    """ Test a complete Fermion wavefunction composed of two Slater determinants. """
    from base_dist import FreeFermion
    nup, ndown = 3, 6
    orbitals_up, Es_up = ho2d.fermion_states_random(nup)
    orbitals_down, Es_down = ho2d.fermion_states_random(ndown)

    freefermion = FreeFermion()
    x = torch.randn(batch, nup + ndown, 2, requires_grad=True)
    logp, grad_logp, laplacian_logp = y_grad_laplacian(
        lambda x: freefermion.log_prob(orbitals_up, orbitals_down, x), x)
    kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))
    potential = 0.5 * (x**2).sum(dim=(-2, -1))
    Eloc = kinetic + potential
    #print("Eloc:", Eloc, "\nEs_up:", Es_up, "Es_down:", Es_down)

    assert torch.allclose(Eloc, sum(Es_up + Es_down) * torch.ones(batch))

def test_FreeFermionHO2D_sample():
    from orbitals import HO2D
    from base_dist import FreeFermion
    ho2d = HO2D()

    nup, ndown = 3, 6
    orbitals_up, _ = ho2d.fermion_states_random(nup)
    orbitals_down, _ = ho2d.fermion_states_random(ndown)

    freefermion = FreeFermion()
    batch, n = (20, 30), nup + ndown
    x = torch.randn(*batch, n, 2)

    logp = freefermion.log_prob(orbitals_up, orbitals_down, x)
    assert logp.shape == batch

    samples = freefermion.sample(orbitals_up, orbitals_down, batch)
    assert samples.shape == (*batch, n, 2)
