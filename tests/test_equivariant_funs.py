import torch
torch.set_default_dtype(torch.float64)

def test_Backflow():
    from src.MLP import MLP
    from src.equivariant_funs import Backflow
    from src.utils import divergence
    import time

    print("\n---- Backflow test ----")
    D_hidden_eta = 100
    eta = MLP(1, D_hidden_eta)
    D_hidden_mu = 200
    mu = MLP(1, D_hidden_mu)
    v = Backflow(eta, mu=mu)

    batch, n, dim = 1000, 10, 3
    x = torch.randn(batch, n, dim, requires_grad=True)
    output = v(x)
    assert output.shape == (batch, n, dim)
    P = torch.randperm(n)
    Px = x[:, P, :]
    Poutput = v(Px)
    assert torch.allclose(Poutput, output[:, P, :])

    start = time.time()
    div = divergence(v, x)
    print("div time:", time.time() - start)

    start = time.time()
    div_direct = v.divergence(x)
    print("div_direct time:", time.time() - start)

    assert div_direct.shape == (batch,)
    assert torch.allclose(div, div_direct)

def test_Backflow_offset():
    """ Test that the two-body backflow transformation yields correct result. """
    from src.MLP import MLP
    from src.equivariant_funs import Backflow

    D_hidden = 100
    eta = MLP(1, D_hidden)
    v = Backflow(eta)

    batch, n, dim = 1000, 10, 3
    x = torch.randn(batch, n, dim)

    output = v._e_e(x)
    def e_e_naive(v, x):
        n = x.shape[-2]
        rij = x[:, :, None] - x[:, None]
        dij = rij.norm(dim=-1, keepdim=True)
        return (v.eta(dij) * rij).sum(dim=-2)
    output_naive = e_e_naive(v, x)

    assert torch.allclose(output, output_naive)
