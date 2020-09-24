import torch
torch.set_default_dtype(torch.float64)

def test_solveivp_function_1():
    """
        A "single ODE" test. See https://github.com/rtqichen/torchdiffeq/issues/29.
        Normal python function version.
    """
    from NeuralODE.function import solve_ivp_function

    def f(t, x, a1, a2):
        return a1 * a2 * x

    a1 = torch.randn(1, requires_grad=True)
    a2 = torch.randn(1, requires_grad=True)

    x0 = torch.randn(3, 4, requires_grad=True)

    t = 0.79
    t_span = (0., t)

    xt = solve_ivp_function(f, t_span, x0, a1, a2)

    loss = xt.sum()
    loss_analytic = x0.sum() * torch.exp(a1 * a2 * t)
    assert torch.allclose(loss, loss_analytic)

    da1, da2, dx0 = torch.autograd.grad(loss, (a1, a2, x0), create_graph=True)
    assert torch.allclose(da1, a2 * t * loss)
    assert torch.allclose(da2, a1 * t * loss)
    assert torch.allclose(dx0, torch.exp(a1 * a2 * t) * torch.ones(3, 4))

    d2a1, = torch.autograd.grad(da1, a1, create_graph=True)
    d2a2, = torch.autograd.grad(da2, a2, create_graph=True)
    assert torch.allclose(d2a1, a2**2 * t**2 * loss)
    assert torch.allclose(d2a2, a1**2 * t**2 * loss)

def test_solveivp_nnmodule_1():
    """
        A "single ODE" test. See https://github.com/rtqichen/torchdiffeq/issues/29.
        torch.nn.Module version.
    """
    from NeuralODE.nnModule import solve_ivp_nnmodule

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            self.a1 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
            self.a2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
            
        def forward(self, t, x):
            return self.a1 * self.a2 * x
    f = F()

    x0 = torch.randn(3, 4, requires_grad=True)

    t = 0.79
    t_span = (0., t)

    xt = solve_ivp_nnmodule(f, t_span, x0)

    loss = xt.sum()
    loss_analytic = x0.sum() * torch.exp(f.a1 * f.a2 * t)
    assert torch.allclose(loss, loss_analytic)

    da1, da2, dx0 = torch.autograd.grad(loss, (f.a1, f.a2, x0), create_graph=True)
    assert torch.allclose(da1, f.a2 * t * loss)
    assert torch.allclose(da2, f.a1 * t * loss)
    assert torch.allclose(dx0, torch.exp(f.a1 * f.a2 * t) * torch.ones(3, 4))

    d2a1, = torch.autograd.grad(da1, f.a1, create_graph=True)
    d2a2, = torch.autograd.grad(da2, f.a2, create_graph=True)
    assert torch.allclose(d2a1, f.a2**2 * t**2 * loss)
    assert torch.allclose(d2a2, f.a1**2 * t**2 * loss)

    xt_onlyxs = solve_ivp_nnmodule(f, t_span, x0, params_require_grad=False)
    loss_onlyxs = xt_onlyxs.sum()
    assert torch.allclose(loss_onlyxs, loss_analytic)
    dx0_onlyxs, = torch.autograd.grad(loss_onlyxs, x0)
    assert torch.allclose(dx0_onlyxs, torch.exp(f.a1 * f.a2 * t) * torch.ones(3, 4))

def test_solveivp_function_2():
    from NeuralODE.function import solve_ivp_function

    def f(t, xs, a1, a2):
        x1, x2 = xs
        return -a1*x1, -a2*x2

    x10 = torch.randn(5, requires_grad=True)
    x20 = torch.randn(3, 4, requires_grad=True)
    a1, a2 = torch.tensor(2.3, requires_grad=True), torch.tensor(1.5, requires_grad=True)

    t = 0.79
    t_span = (0., t)

    x1t, x2t = solve_ivp_function(f, t_span, (x10, x20), a1, a2)

    loss = x1t.sum() + x2t.sum()
    loss_analytic = x10.sum() * torch.exp(-a1*t) + x20.sum() * torch.exp(-a2*t)
    assert torch.allclose(loss, loss_analytic)

    da1, da2, dx10, dx20 = torch.autograd.grad(loss, (a1, a2, x10, x20), create_graph=True)
    assert torch.allclose(da1, -t * x1t.sum())
    assert torch.allclose(da2, -t * x2t.sum())
    assert torch.allclose(dx10, torch.exp(-a1*t) * torch.ones(5))
    assert torch.allclose(dx20, torch.exp(-a2*t) * torch.ones(3, 4))

    d2a1, = torch.autograd.grad(da1, a1, create_graph=True)
    d2a2, = torch.autograd.grad(da2, a2, create_graph=True)
    assert torch.allclose(d2a1, t**2 * x1t.sum())
    assert torch.allclose(d2a2, t**2 * x2t.sum())

    d3a1, = torch.autograd.grad(d2a1, a1, create_graph=True)
    d3a2, = torch.autograd.grad(d2a2, a2, create_graph=True)
    assert torch.allclose(d3a1, -t**3 * x1t.sum())
    assert torch.allclose(d3a2, -t**3 * x2t.sum())

def test_solveivp_nnmodule_2():
    from NeuralODE.nnModule import solve_ivp_nnmodule

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            self.a1 = torch.nn.Parameter(torch.tensor(2.3, requires_grad=True))
            self.a2 = torch.nn.Parameter(torch.tensor(1.5, requires_grad=True))
            
        def forward(self, t, xs):
            x1, x2 = xs
            return -self.a1*x1, -self.a2*x2
    f = F()

    x10 = torch.randn(5, requires_grad=True)
    x20 = torch.randn(3, 4, requires_grad=True)

    t = 0.79
    t_span = (0., t)

    x1t, x2t = solve_ivp_nnmodule(f, t_span, (x10, x20))

    loss = x1t.sum() + x2t.sum()
    loss_analytic = x10.sum() * torch.exp(-f.a1*t) + x20.sum() * torch.exp(-f.a2*t)
    assert torch.allclose(loss, loss_analytic)

    da1, da2, dx10, dx20 = torch.autograd.grad(loss, (f.a1, f.a2, x10, x20), create_graph=True)
    assert torch.allclose(da1, -t * x1t.sum())
    assert torch.allclose(da2, -t * x2t.sum())
    assert torch.allclose(dx10, torch.exp(-f.a1*t) * torch.ones(5))
    assert torch.allclose(dx20, torch.exp(-f.a2*t) * torch.ones(3, 4))

    d2a1, = torch.autograd.grad(da1, f.a1, create_graph=True)
    d2a2, = torch.autograd.grad(da2, f.a2, create_graph=True)
    assert torch.allclose(d2a1, t**2 * x1t.sum())
    assert torch.allclose(d2a2, t**2 * x2t.sum())

    d3a1, = torch.autograd.grad(d2a1, f.a1, create_graph=True)
    d3a2, = torch.autograd.grad(d2a2, f.a2, create_graph=True)
    assert torch.allclose(d3a1, -t**3 * x1t.sum())
    assert torch.allclose(d3a2, -t**3 * x2t.sum())

    x1t_onlyxs, x2t_onlyxs = solve_ivp_nnmodule(f, t_span, (x10, x20), params_require_grad=False)
    loss_onlyxs = x1t_onlyxs.sum() + x2t_onlyxs.sum()
    assert torch.allclose(loss_onlyxs, loss_analytic)
    dx10_onlyxs, dx20_onlyxs = torch.autograd.grad(loss_onlyxs, (x10, x20))
    assert torch.allclose(dx10_onlyxs, torch.exp(-f.a1*t) * torch.ones(5))
    assert torch.allclose(dx20_onlyxs, torch.exp(-f.a2*t) * torch.ones(3, 4))
