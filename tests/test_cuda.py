import torch
torch.set_default_dtype(torch.float64)
import time, pytest

device = torch.device("cuda:2")

@pytest.mark.skipif(not torch.cuda.is_available(), 
        reason="No GPU support in online test envionment")
def test_matmul():
    print("\n---- Test matrix multiplication ----")
    n = 3000
    ntests = 50

    a = torch.randn(n, n)
    b = torch.randn(n, n)

    start = time.time()
    for i in range(ntests):
        ab = a.matmul(b)
        print("CPU %02d" % (i + 1))
    print("Average: ", (time.time() - start) / ntests)

    a_cuda = a.to(device=device)
    b_cuda = b.to(device=device)

    start = time.time()
    for i in range(ntests):
        ab_cuda = a_cuda.matmul(b_cuda)
        print("GPU %02d" % (i + 1))
    print("Average: ", (time.time() - start) / ntests)

@pytest.mark.skipif(not torch.cuda.is_available(), 
        reason="No GPU support in online test envionment")
def test_Backflow():
    from src.MLP import MLP
    from src.equivariant_funs import Backflow
    print("\n---- Test Backflow network ----")

    D_hidden = 200
    batch, n, dim = 10000, 4, 2
    ntests = 50

    eta = MLP(1, D_hidden)
    v = Backflow(eta)
    x = torch.randn(batch, n, dim)

    start = time.time()
    for i in range(ntests):
        value, div = v(x), v.divergence(x)
        print("CPU %02d" % (i + 1))
    print("Average: ", (time.time() - start) / ntests)

    assert value.shape == (batch, n, dim)
    assert div.shape == (batch,)

    v_cuda = v.to(device=device)
    x_cuda = x.to(device=device)

    start = time.time()
    for i in range(ntests):
        value_cuda, div_cuda = v_cuda(x_cuda), v_cuda.divergence(x_cuda)
        print("GPU %02d" % (i + 1))
    print("Average: ", (time.time() - start) / ntests)

    assert torch.allclose(value, value_cuda.to(device="cpu"))
    assert torch.allclose(div, div_cuda.to(device="cpu"))
