import torch
torch.set_default_dtype(torch.float64)

from src.MLP import MLP

def test_reproducibility():
    """ Test the initial parameters are the same by setting the same seed. """
    dim, D_hidden = 15, 40
    seed = 42
    mlp1 = MLP(dim, D_hidden)
    mlp1.init_gaussian(seed)
    mlp2 = MLP(dim, D_hidden)
    mlp2.init_gaussian(seed)
    assert torch.allclose(mlp1.fc1.weight, mlp2.fc1.weight)
    assert torch.allclose(mlp1.fc1.bias, mlp2.fc1.bias)
    assert torch.allclose(mlp1.fc2.weight, mlp2.fc2.weight)

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
