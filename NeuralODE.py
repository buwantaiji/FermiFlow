import torch
torch.set_default_dtype(torch.float64)

from scipy.integrate import solve_ivp

def _flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])

def _shapes_numels(tensors):
    return tuple((tensor.shape, tensor.numel()) for tensor in tensors)

def _unflatten(tensor, shapes_numels):
    baseidx = 0
    unflatten_tensors = []
    for shape, numel in shapes_numels:
        unflatten_tensors.append(tensor[baseidx:(baseidx+numel)].reshape(shape))
        baseidx += numel
    return tuple(unflatten_tensors)

class SolveIVP(torch.autograd.Function):
    """ Kernel primitive of the NeuralODE solver. """
    @staticmethod
    def forward(ctx, f, t_span, *x0s_None_params_None_others):

        """
            ---INPUT---

            f: the function in the right hand side of the ODE. f can be either a
               torch.nn.Module or a normal python function, with corresponding
               signatures as follows:

               torch.nn.Module: f(t, xs)
               python function: f(t, xs, *params)

               In this primitive, xs MUST be a tuple containing several tensor
               variables; params is a tuple containing the parameter tensors. 
               The output of f is another tuple of tensors, which should contain
               exactly the same number of tensors and corresponding shapes with xs.

            t_span: a 2-tuple of floats (T0, T) representing the interval of integration.

            x0s_None_params_None_others: a tuple containing a sequence of initial 
                condition tensors (x0s), followed by a delimiter of value None, 
                followed by a sequence of given parameter tensors, followed by 
                another delimiter None, followed by the hyperparameters rtol and atol.

            ---OUTPUT---

            The tensor xs at time T, given the initial condition x0s at time T0, by
            integrating the ODE at the given parameter values. As expected, xs must be
            a tuple of tensors, and have exactly the same number of tensors and
            corresponding shapes with xs in the signature of f.
        """

        delimiter_gen = (idx for idx, val in enumerate(x0s_None_params_None_others) if val is None)

        delimiter1 = next(delimiter_gen)
        delimiter2 = next(delimiter_gen)
        x0s, params, others = x0s_None_params_None_others[:delimiter1], \
                              x0s_None_params_None_others[delimiter1+1:delimiter2], \
                              x0s_None_params_None_others[delimiter2+1:]
        rtol, atol = others
        xs_shapes_numels, params_shapes_numels = _shapes_numels(x0s), _shapes_numels(params)

        def f_wrapper(t, x_flatten):
            xs = _unflatten(torch.from_numpy(x_flatten), xs_shapes_numels)
            outputs = f(t, xs) if isinstance(f, torch.nn.Module) else f(t, xs, *params)
            output_flatten = _flatten(outputs).numpy()
            return output_flatten

        x0_flatten = _flatten(x0s).numpy()
        print("rtol:", rtol, "atol:", atol)
        sol = solve_ivp(f_wrapper, t_span, x0_flatten, t_eval=t_span[-1:], 
                        rtol=rtol, atol=atol)
        xts = _unflatten( torch.from_numpy(sol.y[:, -1]), xs_shapes_numels )

        ctx.save_for_backward(*xts, *params)
        ctx.f, ctx.t_span, ctx.xs_shapes_numels, ctx.params_shapes_numels = \
                f, t_span, xs_shapes_numels, params_shapes_numels
        ctx.rtol, ctx.atol = rtol, atol
        return xts

    @staticmethod
    def backward(ctx, *grad_xts):
        f, t_span, xs_shapes_numels, params_shapes_numels = \
                ctx.f, ctx.t_span, ctx.xs_shapes_numels, ctx.params_shapes_numels
        rtol, atol = ctx.rtol, ctx.atol
        n_xs, n_params = len(xs_shapes_numels), len(params_shapes_numels)
        xts, params = ctx.saved_tensors[:n_xs], ctx.saved_tensors[n_xs:]

        f_aug = augmented_dynamics(f, xs_shapes_numels, params_shapes_numels)

        t_span = t_span[1], t_span[0]
        adjoint_params0 = tuple(torch.zeros(shape) for shape, _ in params_shapes_numels)
        x_aug0 = xts + grad_xts + adjoint_params0

        if isinstance(f_aug, torch.nn.Module):
            xt_aug = solve_ivp_nnmodule(f_aug, t_span, x_aug0, 
                                        rtol=rtol, atol=atol)
        else:
            xt_aug = solve_ivp_function(f_aug, t_span, x_aug0, *params, 
                                        rtol=rtol, atol=atol)

        _, adjoint_x0s, adjoint_params = \
            xt_aug[:n_xs], xt_aug[n_xs:2*n_xs], xt_aug[2*n_xs:]

        return (None, None, *adjoint_x0s, None, *adjoint_params, None, None, None)

def augmented_dynamics(f, xs_shapes_numels, params_shapes_numels):
    n_xs, n_params = len(xs_shapes_numels), len(params_shapes_numels)

    if isinstance(f, torch.nn.Module):

        """ f is a torch.nn.Module. """

        class F_aug(torch.nn.Module):
            def __init__(self, f):
                super(F_aug, self).__init__()
                """
                    F_aug takes f as its submodule, thus inheritates all the
                registered parameters.
                """
                self.f = f
                self.f_params = tuple(self.f.parameters())

            def forward(self, t, x_aug):
                xs, adjoint_xs, adjoint_params = x_aug[:n_xs], x_aug[n_xs:2*n_xs], x_aug[2*n_xs:]

                with torch.enable_grad():
                    xs = tuple(x.requires_grad_(True) for x in xs)
                    adjoint_xs = tuple(adjoint_x.requires_grad_(True) for adjoint_x in adjoint_xs)
                    adjoint_params = tuple(adjoint_param.requires_grad_(True) for adjoint_param in adjoint_params)

                    f_values = self.f(t, xs)
                    forward_value = -sum((adjoint_x * f_value).sum() for adjoint_x, f_value in zip(adjoint_xs, f_values))
                    vjp_xs_params = torch.autograd.grad(forward_value, 
                            xs + self.f_params, create_graph=True, allow_unused=True)

                    vjp_xs, vjp_params = vjp_xs_params[:n_xs], vjp_xs_params[n_xs:]
                    vjp_xs = tuple(vjp_x if vjp_x is not None else torch.zeros(shape) 
                                for vjp_x, (shape, _) in zip(vjp_xs, xs_shapes_numels))

                    return f_values + vjp_xs + vjp_params
        f_aug = F_aug(f)
    else:

        """ f is a normal python function. """

        def f_aug(t, x_aug, *params):
            xs, adjoint_xs, adjoint_params = x_aug[:n_xs], x_aug[n_xs:2*n_xs], x_aug[2*n_xs:]

            with torch.enable_grad():
                xs = tuple(x.requires_grad_(True) for x in xs)
                adjoint_xs = tuple(adjoint_x.requires_grad_(True) for adjoint_x in adjoint_xs)
                adjoint_params = tuple(adjoint_param.requires_grad_(True) for adjoint_param in adjoint_params)

                params = tuple(param.requires_grad_(True) for param in params)

                f_values = f(t, xs, *params)
                forward_value = -sum((adjoint_x * f_value).sum() for adjoint_x, f_value in zip(adjoint_xs, f_values))
                vjp_xs_params = torch.autograd.grad(forward_value, 
                        xs + params, create_graph=True, allow_unused=True)

                vjp_xs, vjp_params = vjp_xs_params[:n_xs], vjp_xs_params[n_xs:]
                vjp_xs = tuple(vjp_x if vjp_x is not None else torch.zeros(shape) 
                            for vjp_x, (shape, _) in zip(vjp_xs, xs_shapes_numels))

                return f_values + vjp_xs + vjp_params
    return f_aug


def solve_ivp_function(f, t_span, x0s, *params, rtol=1e-6, atol=1e-8):
    if not callable(f) or (callable(f) and isinstance(f, torch.nn.Module)):
        raise ValueError("f is required to be a normal python function.")

    if isinstance(x0s, torch.Tensor):

        """ x0s is a torch.Tensor. """

        def f_wrapper(t, xs, *params):
            return (f(t, xs[0], *params),)

        return SolveIVP.apply(f_wrapper, t_span, x0s, None, *params, 
                            None, rtol, atol)[0]
    else:

        """ x0s is a tuple of several torch.Tensors."""

        return SolveIVP.apply(f, t_span, *x0s, None, *params, 
                            None, rtol, atol)
    

def solve_ivp_nnmodule(f, t_span, x0s, rtol=1e-6, atol=1e-8):
    if not isinstance(f, torch.nn.Module):
        raise ValueError("f is required to be an instance of torch.nn.Module.")

    if isinstance(x0s, torch.Tensor):

        """ x0s is a torch.Tensor. """

        class F_wrapper(torch.nn.Module):
            def __init__(self, f):
                super(F_wrapper, self).__init__()
                self.f = f
            def forward(self, t, xs):
                return (self.f(t, xs[0]),)

        f_wrapper = F_wrapper(f)
        params = f_wrapper.parameters()
        return SolveIVP.apply(f_wrapper, t_span, x0s, None, *params, 
                            None, rtol, atol)[0]
    else:

        """ x0s is a tuple of several torch.Tensors."""

        params = f.parameters()
        return SolveIVP.apply(f, t_span, *x0s, None, *params, 
                            None, rtol, atol)
