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

def test_odeint_cnf():
    from MLP import MLP
    from equivariant_funs import Backflow
    from NeuralODE.nnModule import solve_ivp_nnmodule
    import time
    print("\n---- Comparison of various ODE solvers ----")

    D_hidden = 200
    batch, n, dim = 1000, 4, 2

    eta = MLP(1, D_hidden)
    v = Backflow(eta)

    class V_wrapper(torch.nn.Module):
        def __init__(self, v):
            super(V_wrapper, self).__init__()
            self.v = v
        def forward(self, t, x):
            return self.v(x)
    v_wrapper = V_wrapper(v)

    class F(torch.nn.Module):
        def __init__(self, v):
            super(F, self).__init__()
            self.v = v
        def forward(self, t, x_and_logp):
            x, _ = x_and_logp
            return self.v(x), -self.v.divergence(x)
    f = F(v)

    t_span = (0., 1.)
    t_span_inverse = t_span[1], t_span[0]

    from torch.distributions.normal import Normal
    from torch.distributions.independent import Independent
    dist = Normal(torch.zeros(n, dim), (0.5 * torch.ones(n, dim)).sqrt())
    dist = Independent(dist, reinterpreted_batch_ndims=2)

    ntests = 20

    z = dist.sample((batch,))

    def scipy_time(f, t_span, x0s, ntests):
        start = time.time()
        for i in range(ntests):
            x = solve_ivp_nnmodule(f, t_span, x0s, params_require_grad=False)
            print("%02d" % (i+1))
        print("Average: ", (time.time() - start) / ntests)
        return x

    def odeint_time(f, t_span, x0s, ntests):
        start = time.time()
        for i in range(ntests):
            xs = odeint(f, x0s, torch.tensor(t_span), rtol=1e-6, atol=1e-8)
            x = xs[-1] if isinstance(xs, torch.Tensor) else tuple(x[-1] for x in xs)
            print("%02d" % (i+1))
        print("Average: ", (time.time() - start) / ntests)
        return x

    def odeint_time_cuda(f, t_span, x0s, device, ntests):
        start = time.time()
        for i in range(ntests):
            xs = odeint(f, x0s, torch.tensor(t_span, device=device), rtol=1e-6, atol=1e-8)
            x = xs[-1] if isinstance(xs, torch.Tensor) else tuple(x[-1] for x in xs)
            print("%02d" % (i+1))
        print("Average: ", (time.time() - start) / ntests)
        return x

    print("---- Call in scipy ----")
    print("1. Single ODE")
    x = scipy_time(v_wrapper, t_span, z, ntests)
    print("2. Two ODEs")
    z_original, delta_logp = scipy_time(f, t_span_inverse, (x, torch.zeros(batch)), ntests)
    logp = dist.log_prob(z_original) - delta_logp

    print("---- odeint from torchdiffeq ----")
    print("1. Single ODE")
    x_odeint = odeint_time(v_wrapper, t_span, z, ntests)
    print("2. Two ODEs")
    z_odeint, delta_logp_odeint = odeint_time(f, t_span_inverse, 
                        (x_odeint, torch.zeros(batch)), ntests)
    logp_odeint = dist.log_prob(z_odeint) - delta_logp_odeint

    assert torch.allclose(x_odeint, x, atol=5e-7)
    #assert torch.allclose(logp_odeint, logp, atol=5e-4)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        v_wrapper_cuda = v_wrapper.to(device=device)
        f_cuda = f.to(device=device)
        z_cuda = z.to(device=device)
        dist_cuda = Normal(torch.zeros(n, dim, device=device), 
                                    (0.5 * torch.ones(n, dim, device=device)).sqrt())
        dist_cuda = Independent(dist_cuda, reinterpreted_batch_ndims=2)

        print("---- odeint from torchdiffeq, with GPU ----")
        print("1. Single ODE")
        x_odeint_cuda = odeint_time_cuda(v_wrapper_cuda, t_span, z_cuda, device, ntests)
        print("2. Two ODEs")
        z_odeint_cuda, delta_logp_odeint_cuda = odeint_time_cuda(f_cuda, t_span_inverse, 
                        (x_odeint_cuda, torch.zeros(batch, device=device)), device, ntests)
        logp_odeint_cuda = dist_cuda.log_prob(z_odeint_cuda) - delta_logp_odeint_cuda

        assert torch.allclose(x_odeint_cuda.to(device="cpu"), x_odeint)
        assert torch.allclose(logp_odeint_cuda.to(device="cpu"), logp_odeint, atol=5e-7)
