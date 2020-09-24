import torch
torch.set_default_dtype(torch.float64)

from NeuralODE_nnModule import solve_ivp_nnmodule

class CNF(torch.nn.Module):
    def __init__(self, basedist, v, t_span, pair_potential, sp_potential=None):
        """
            basedist: base distribution, which is an instance of the subclass of BaseDist.

            v: callable representing the vector-valued function v in the r.h.s of the ODE.
               v MUST be an instance of torch.nn.Module. 
               Also note that the calling signature of v is default to be v(x), 
               i.e., without making use of the time variable t.

            t_span: 2-tuple of floats (T0, T) representing the interval of integration.
        """
        super(CNF, self).__init__()
        self.basedist = basedist

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
        self.t_span_inverse = t_span[1], t_span[0]

        self.pair_potential = pair_potential
        self.sp_potential = sp_potential

    def sample(self, sample_shape):
        z = self.basedist.sample(sample_shape)
        x = solve_ivp_nnmodule(self.v_wrapper, self.t_span, z, params_require_grad=False)
        return z, x

    def logp(self, x, params_require_grad=False):
        batch = x.shape[0]
        z, delta_logp = solve_ivp_nnmodule(self.f, self.t_span_inverse, 
                    (x, torch.zeros(batch)), params_require_grad=params_require_grad)
        logp = self.basedist.log_prob(z) - delta_logp
        return logp

    def check_reversibility(self, batch):
        z, x = self.sample((batch,))
        _, logp = solve_ivp_nnmodule(self.f, self.t_span, (z, self.basedist.log_prob(z)), 
                                        params_require_grad=False)
        z_reverse, delta_logp = solve_ivp_nnmodule(self.f, self.t_span_inverse, 
                        (x, torch.zeros(batch)), params_require_grad=False)
        logp_reverse = self.basedist.log_prob(z_reverse) - delta_logp

        print("MaxAbs of z_reverse - z:", (z_reverse - z).abs().max())
        #print(logp, logp_reverse)
        print("logp_reverse - logp:", logp_reverse - logp)
        print("MaxAbs of logp_inverse - logp:", (logp_reverse - logp).abs().max())

    def forward(self, batch):
        from utils import y_grad_laplacian

        z, x = self.sample((batch,))
        x.requires_grad_(True)

        logp_full = self.logp(x, params_require_grad=True)

        logp, grad_logp, laplacian_logp = y_grad_laplacian(self.logp, x) 
        kinetic = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(dim=(-2, -1))

        potential = self.pair_potential.V(x)
        if self.sp_potential:
            potential += self.sp_potential.V(x)

        Eloc = kinetic + potential

        self.E = Eloc.detach().mean().item()
        gradE = (logp_full * (Eloc.detach() - self.E)).mean()
        return gradE


if __name__ == "__main__":
    from base_dist import FreeBosonHO

    from MLP import MLP
    from equivariant_funs import Backflow
    #from equivariant_funs import FermiNet

    from potentials import HO, GaussianPairPotential

    n, dim = 4, 2

    freebosonho = FreeBosonHO(n, dim)

    D_hidden = 100
    eta = MLP(1, D_hidden)
    v = Backflow(eta)
    #L, spsize, tpsize = 2, 16, 8
    #v = FermiNet(n, dim, L, spsize, tpsize)

    t_span = (0., 1.)

    sp_potential = HO()
    g, s = 9.0, 0.5
    pair_potential = GaussianPairPotential(g, s)

    cnf = CNF(freebosonho, v, t_span, pair_potential, sp_potential=sp_potential)
    
    batch = 8000
    #cnf.check_reversibility(batch)

    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-2)
    iter_num = 200
    print("iter_num:", iter_num)
    for i in range(iter_num):
        gradE = cnf(batch)
        optimizer.zero_grad()
        gradE.backward()
        gradE = optimizer.step()
        print("iter: %02d" % i, "Energy:", cnf.E)
