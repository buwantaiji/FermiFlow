"""
    Some simple tests of the ODE solver odeint implemented in torchdiffeq.
    See https://github.com/rtqichen/torchdiffeq for details.
"""
import torch
torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

def test_odeint_1():

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            self.a1 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
            self.a2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
            
        def forward(self, t, x):
            return self.a1 * self.a2 * x
    f = F()

    x0 = torch.randn(3, 4)

    t = 0.79
    t_span = (0., t)

    xt = odeint(f, x0, torch.tensor(t_span))[-1]

    xt_analytic = x0 * torch.exp(f.a1 * f.a2 * t)
    assert torch.allclose(xt, xt_analytic)

def test_odeint_2():

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            self.a1 = torch.nn.Parameter(torch.tensor(2.3, requires_grad=True))
            self.a2 = torch.nn.Parameter(torch.tensor(1.5, requires_grad=True))
            
        def forward(self, t, xs):
            x1, x2 = xs
            return -self.a1*x1, -self.a2*x2
    f = F()

    x10 = torch.randn(5)
    x20 = torch.randn(3, 4)

    t = 0.79
    t_span = (0., t)

    xts = odeint(f, (x10, x20), torch.tensor(t_span))
    x1t, x2t = tuple(xt[-1] for xt in xts)

    x1t_analytic = x10 * torch.exp(-f.a1*t)
    x2t_analytic = x20 * torch.exp(-f.a2*t)
    assert torch.allclose(x1t, x1t_analytic)
    assert torch.allclose(x2t, x2t_analytic)
