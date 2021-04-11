import torch
torch.set_default_dtype(torch.float64)

from NeuralODE.nnModule import solve_ivp_nnmodule

class CNF(torch.nn.Module):
    def __init__(self, v, t_span):
        """
            v: callable representing the vector-valued function v in the r.h.s of the ODE.
               v MUST be an instance of torch.nn.Module. 
               Also note that the calling signature of v is default to be v(x), 
               i.e., without making use of the time variable t.

            t_span: 2-tuple of floats (T0, T) representing the interval of integration.
        """
        super(CNF, self).__init__()

        class V_wrapper(torch.nn.Module):
            """
                A simple wrapper of the equivariant function to meet the signature
            used in the ODE solver.
            """
            def __init__(self, v):
                super(V_wrapper, self).__init__()
                self.v = v
            def forward(self, t, x):
                return self.v(x)
        self.v_wrapper = V_wrapper(v)

        class F(torch.nn.Module):
            def __init__(self, v):
                super(F, self).__init__()
                self.v = v
            def forward(self, t, x_and_logp):
                x, _ = x_and_logp
                return self.v(x), -self.v.divergence(x)
        self.f = F(v)

        self.t_span = t_span
        self.t_span_reverse = t_span[1], t_span[0]

    def generate(self, z, nframes=None):
        if nframes is None:
            x = solve_ivp_nnmodule(self.v_wrapper, self.t_span, z, params_require_grad=False)
        else:
            from torchdiffeq import odeint
            t = torch.linspace(*self.t_span, steps=nframes, device=z.device)
            x = odeint(self.v_wrapper, z, t)
        return x

    def delta_logp(self, x, params_require_grad=False):
        batch = x.shape[0]
        z, delta_logp = solve_ivp_nnmodule(self.f, self.t_span_reverse, 
                (x, torch.zeros(batch, device=x.device)), params_require_grad=params_require_grad)
        return z, delta_logp

    def check_reversibility(self, basedist, batch):
        print("---- CNF REVERSIBILITY CHECK ----")
        z = basedist.sample((batch,))
        x = self.generate(z)
        _, logp = solve_ivp_nnmodule(self.f, self.t_span, (z, basedist.log_prob(z)), 
                                        params_require_grad=False)
        z_reverse, delta_logp = self.delta_logp(x)
        logp_reverse = basedist.log_prob(z_reverse) - delta_logp

        print("MaxAbs of z_reverse - z:", (z_reverse - z).abs().max())
        #print("logp_reverse:", logp_reverse, "\nlogp:", logp)
        print("logp_reverse - logp:", logp_reverse - logp)
        print("MaxAbs of logp_inverse - logp:", (logp_reverse - logp).abs().max())

    def backflow_potential(self):
        from equivariant_funs import Backflow
        if not isinstance(self.v_wrapper.v, Backflow):
            raise TypeError("The underlying equivariant transformation is not Backflow.")
        eta = self.v_wrapper.v.eta
        mu = self.v_wrapper.v.mu
        return eta, mu

if __name__ == "__main__":
    """ Perform the reversibility check. """
    from MLP import MLP
    from equivariant_funs import Backflow
    from base_dist import FreeBosonHO

    D_hidden_eta = 50
    eta = MLP(1, D_hidden_eta)
    v = Backflow(eta, mu=None)

    t_span = (0., 1.)

    n, dim = 4, 2
    device = torch.device("cuda:1")
    basedist = FreeBosonHO(n, dim, device=device)

    cnf = CNF(v, t_span)
    cnf.to(device=device)

    batch = 200
    cnf.check_reversibility(basedist, batch)
