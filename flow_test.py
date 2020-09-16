import torch
torch.set_default_dtype(torch.float64)

from NeuralODE import solve_ivp_nnmodule

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

        class F(torch.nn.Module):
            def __init__(self, v):
                super(F, self).__init__()
                self.v = v
            def forward(self, t, x_and_logp):
                from utils import divergence
                with torch.enable_grad():
                    x, _ = x_and_logp
                    x.requires_grad_(True)
                    return self.v(x), - divergence(self.v, x, create_graph=True)
        self.f = F(v)

        self.t_span = t_span
        self.t_span_inverse = t_span[1], t_span[0]

        from torch.distributions.normal import Normal
        from torch.distributions.independent import Independent
        self.dist = Normal(torch.zeros(n, dim), (0.5 * torch.ones(n, dim)).sqrt())
        self.dist = Independent(self.dist, reinterpreted_batch_ndims=2)

    def logp(self, x):
        batch = x.shape[0]
        z, delta_logp = solve_ivp_nnmodule(self.f, self.t_span_inverse, 
                        (x, torch.zeros(batch)))
        logp = self.dist.log_prob(z) - delta_logp
        return logp

    def forward(self, batch):
        from utils import y_grad_laplacian

        x = torch.randn(batch, n, dim, requires_grad=True)

        logp = self.logp(x)
        print("Computed logp.")

        f_value_x, f_value_logp = self.f(None, (x, logp))
        adjoint_x, adjoint_logp = torch.randn(batch, n, dim), torch.randn(batch)
        forward_value = - ( (adjoint_x*f_value_x).sum() + (adjoint_logp*f_value_logp).sum() )
        vjp_x, vjp_logp, *vjp_params = torch.autograd.grad(forward_value, 
                                (x, logp, *tuple(self.f.parameters())), allow_unused=True)
        print("vjp_x:", vjp_x)
        print("vjp_logp:", vjp_logp)
        print("vjp_params:", vjp_params)
        #grad_logp, = torch.autograd.grad(logp, x, grad_outputs=torch.ones(batch))
        #print("Computed grad_logp.")

if __name__ == "__main__":
    #from equivariant_funs import FermiNet
    from drift import Drift

    n, dim = 5, 2

    L, spsize, tpsize = 2, 16, 8
    #v = FermiNet(n, dim, L, spsize, tpsize)
    v = Drift(L, spsize, tpsize, n, dim)
    v.set_time(0.0)

    t_span = (0., 1.)

    cnf = CNF(v, t_span)
    
    batch = 100
    cnf.forward(batch)
