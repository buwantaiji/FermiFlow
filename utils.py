import torch
torch.set_default_dtype(torch.float64)

def divergence(v, x, create_graph=False):
    """
        Compute the "batch-wise" divergence of the vector-valued function v respect to x.

        x: (batch, ...),  v(x): (batch, ...).
        "..." represents any number of dimensions corresponding to the feature space.

        Output of this function: (batch,)
    """
    x_flatten = x.flatten(start_dim=1)
    y = v(x_flatten.view_as(x))
    y_flatten = y.flatten(start_dim=1)
    batch, dim = x_flatten.shape
    div = sum( torch.autograd.grad(y_flatten[:, i], x_flatten, 
            grad_outputs=torch.ones(batch), 
            retain_graph=True, create_graph=create_graph)[0][:, i] 
            for i in range(dim) )
    return div

def divergence_2(f, xs, create_graph=False):
    """
        Another implementation of divergence from
        https://code.itp.ac.cn/wanglei/fermiflow/-/blob/master/src/utils.py#L14.
    """
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    ys = f(xs_flat.view_as(xs))
    ys_flat = ys.flatten(start_dim=1)
    div_ys = sum(
        torch.autograd.grad(
            yi, xi, torch.ones_like(yi), retain_graph=True, create_graph=create_graph
        )[0]
        for xi, yi in zip(xis, (ys_flat[..., i] for i in range(len(xis))))
    )
    return div_ys

def y_grad_laplacian(f, x):
    """
        Compute the "batch-wise" value of the scalar-valued function f at point x, 
        together with corresponding gradients and laplacians.

        x: (batch, ...),  f(x): (batch).
        "..." represents any number of dimensions corresponding to the feature space.

        Output of this function is a tuple (y, grad_y, laplacian_y), with shapes as follows:
        y: (batch),  grad_y: (batch, ...),  laplacian_y: (batch)
    """
    x_flatten = x.flatten(start_dim=1)
    y = f(x_flatten.view_as(x))
    batch, dim = x_flatten.shape
    print("Computed logp.")

    grad_y_flatten, = torch.autograd.grad(y, x_flatten, 
                            grad_outputs=torch.ones(batch), create_graph=True)
    grad_y = grad_y_flatten.view_as(x)
    print("Computed grad_logp.")

    laplacian_y = sum( torch.autograd.grad(grad_y_flatten[:, i], x_flatten, 
                        grad_outputs=torch.ones(batch), retain_graph=True)[0][:, i]
                        for i in range(dim) )
    print("Computed laplacian_logp.")
    return y, grad_y, laplacian_y
