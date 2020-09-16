import torch
torch.set_default_dtype(torch.float64)

def test_divergence_1():
    from utils import divergence

    def v(x):
        return x**2

    def divergence_v(x):
        return (2*x).flatten(start_dim=1).sum(dim=1)

    batch, n, dim = 10, 5, 3
    x = torch.randn(batch, n, dim, requires_grad=True)
    div = divergence(v, x)
    div_analytic = divergence_v(x)
    assert torch.allclose(div, div_analytic)

def test_divergence_2():
    from utils import divergence

    batch, dim = 20, 30
    W = torch.randn(dim, dim)
    b = torch.randn(dim)

    def linear(x):
        return x.matmul(W) + b

    def divergence_linear(x):
        return W.trace().expand(batch)

    x = torch.randn(batch, dim, requires_grad=True)
    div = divergence(linear, x)
    div_analytic = divergence_linear(x)
    assert torch.allclose(div, div_analytic)

def test_divergence_FermiNet():
    from utils import divergence
    from equivariant_funs import FermiNet

    batch, n, dim = 100, 10, 3
    L, spsize, tpsize = 6, 7, 15
    
    v = FermiNet(n, dim, L, spsize, tpsize)
    x = torch.randn(batch, n, dim, requires_grad=True)
    div = divergence(v, x)
    assert div.shape == (batch,)

    P = torch.randperm(n)
    Px = x[:, P, :]
    Pdiv = divergence(v, Px)
    assert torch.allclose(div, Pdiv)

def test_divergences_time_FermiNet():
    """
        Test the performance of the two implementations of computing divergence.
    """
    from utils import divergence, divergence_2
    from equivariant_funs import FermiNet
    import time

    batch, n, dim = 200, 10, 2
    L, spsize, tpsize = 2, 16, 8
    
    v = FermiNet(n, dim, L, spsize, tpsize)
    x = torch.randn(batch, n, dim, requires_grad=True)

    ntest = 50

    start = time.time()
    for i in range(ntest):
        div1 = divergence(v, x)
        print("Computing divergence: %02d" % (i + 1))
    end = time.time()
    print("time1: ", (end - start) / ntest)

    start = time.time()
    for i in range(ntest):
        div2 = divergence_2(v, x)
        print("Computing divergence_2: %02d" % (i + 1))
    end = time.time()
    print("time2: ", (end - start) / ntest)

    assert torch.allclose(div1, div2)

def test_y_grad_laplacian():
    from utils import y_grad_laplacian

    batch, n, dim = 10, 5, 3
    w = torch.randn(batch, n, dim)

    def f(x):
        return ((x**3 + 5*x**2)*w).sum(dim=(-2, -1))

    def grad_f(x):
        return (3*x**2 + 10*x)*w

    def laplacian_f(x):
        return ((6*x + 10)*w).sum(dim=(-2, -1))

    x = torch.randn(batch, n, dim, requires_grad=True)
    y, grad_y, laplacian_y = y_grad_laplacian(f, x)
    y_analytic, grad_y_analytic, laplacian_y_analytic = \
            f(x), grad_f(x), laplacian_f(x)
    assert torch.allclose(y, y_analytic)
    assert torch.allclose(grad_y, grad_y_analytic)
    assert torch.allclose(laplacian_y, laplacian_y_analytic)
