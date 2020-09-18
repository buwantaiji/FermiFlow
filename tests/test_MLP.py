import torch
torch.set_default_dtype(torch.float64)

from MLP import MLP

def test_batchdim_1():
    """ batch dimension is 1. """
    dim, D_hidden = 15, 40
    mlp = MLP(dim, D_hidden)

    batch = 100
    x = torch.randn(batch, dim, requires_grad=True)
    y = mlp(x)
    assert y.shape == (batch, 1)
    grad_x, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
    grad_x_direct = mlp.grad(x)
    assert grad_x.shape == (batch, dim)
    assert torch.allclose(grad_x, grad_x_direct)

def test_batchdim_2():
    """ batch dimension is 2. """
    dim, D_hidden = 15, 40
    mlp = MLP(dim, D_hidden)

    batch = 46, 87
    x = torch.randn(*batch, dim, requires_grad=True)
    y = mlp(x)
    assert y.shape == (*batch, 1)
    grad_x, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
    grad_x_direct = mlp.grad(x)
    assert grad_x.shape == (*batch, dim)
    assert torch.allclose(grad_x, grad_x_direct)
