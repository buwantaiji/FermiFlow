import torch
torch.set_default_dtype(torch.float64)

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
    from slater import LogAbsSlaterDet, logabsslaterdet
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

def test_LogAbsSlaterDetMultStates():
    """ IMPORTANT TEST
        Test the result of backward of the LogAbsSlaterDetMultStates primitive is 
    consistent with the approach logabsslaterdetmultstates, which works by directly 
    backwarding through the torch.slogdet function.
    """
    from orbitals import HO2D
    from slater import LogAbsSlaterDetMultStates, logabsslaterdetmultstates
    log_abs_slaterdet_multstates = LogAbsSlaterDetMultStates.apply
    from utils import y_grad_laplacian
    import random

    ho2d = HO2D()
    n = 5
    Nstates = 10
    states = tuple(ho2d.fermion_states_random(n)[0] for _ in range(Nstates))
    state_indices_collection = dict(zip(range(Nstates), random.choices(range(5, 20), k=Nstates)))
    batch = sum(times for times in state_indices_collection.values())
    print("state_indices_collection:", state_indices_collection)
    print("batch:", batch)

    x = torch.randn(batch, n, 2, requires_grad=True)

    logabsdet, grad_logabsdet, laplacian_logabsdet= y_grad_laplacian(
            lambda x: log_abs_slaterdet_multstates(states, state_indices_collection, x), x)
    assert logabsdet.shape == (batch,)
    assert grad_logabsdet.shape == (batch, n, 2)
    assert laplacian_logabsdet.shape == (batch,)

    logabsdet_direct, grad_logabsdet_direct, laplacian_logabsdet_direct = y_grad_laplacian(
            lambda x: logabsslaterdetmultstates(states, state_indices_collection, x), x)
    assert torch.allclose(logabsdet_direct, logabsdet)
    assert torch.allclose(grad_logabsdet_direct, grad_logabsdet)
    assert torch.allclose(laplacian_logabsdet_direct, laplacian_logabsdet)
