import torch
torch.set_default_dtype(torch.float64)

class LogAbsSlaterDet(torch.autograd.Function):
    """
        Compute the logarithm of absolute value of a Slater determinant, given
    some orbitals and coordinate vectors.
        The backward of this primitive makes use of the specific structure of
    Slater determinants, which would be more stable in some cases than the more
    general approach.
    """
    @staticmethod
    def forward(ctx, orbitals, x):
        """
            --- INPUT ---

            orbitals: a tuple of length n containing the n orbitals {\phi_i(r)}, 
            each of which is represented by a normal function.
            
            x: the particle coordinates. Generally, x has shape (*batch, n, dim),
               where several batch dimensions are allowed.

            --- OUTPUT ---

            The nxn Slater determinants det(\phi_j(r_i)), which has shape (*batch)
            in the general case.
        """
        *batch, n, _ = x.shape
        D = torch.empty(*batch, n, n, device=x.device)
        for i in range(n):
            D[..., i] = orbitals[i](x)

        ctx.save_for_backward(x)
        ctx.orbitals = orbitals

        _, logabsdet = D.slogdet()
        return logabsdet
    
    @staticmethod
    def backward(ctx, grad_logabsdet):
        x, = ctx.saved_tensors
        orbitals = ctx.orbitals
        *batch, n, dim = x.shape

        with torch.enable_grad():
            """
                Here in backward, it seems that the Slater matrix has to be created
            again to guarantee the correctness of the implementation, especially for
            higher-order gradients. WHY?
            """
            D = torch.empty(*batch, n, n, device=x.device)
            for i in range(n):
                D[..., i] = orbitals[i](x)

            dorbitals = torch.empty(*batch, n, dim, n, device=x.device)
            for i in range(n):
                orbital_value = orbitals[i](x)
                dorbitals[..., i], = torch.autograd.grad(orbital_value, x, 
                        grad_outputs=torch.ones_like(orbital_value), create_graph=True)
            dlogabsdet = torch.einsum("...ndj,...jn->...nd", dorbitals, D.inverse())
            grad_x = grad_logabsdet[..., None, None] * dlogabsdet
            return None, grad_x

def logabsslaterdet(orbitals, x):
    """
        The "straight-forward" version of LogAbsSlaterDet, where the backward is
    taken care of automatically by the torch.slogdet function.
    """
    *batch, n, _ = x.shape
    D = torch.empty(*batch, n, n, device=x.device)
    for i in range(n):
        D[..., i] = orbitals[i](x)
    _, logabsdet = D.slogdet() 
    return logabsdet

class LogAbsSlaterDetMultStates(torch.autograd.Function):
    """
        Compute the logarithm of absolute value of multiple Slater determinants, given
    some many-body states (each correponding to several orbitals) and coordinate vectors.
        The backward of this primitive makes use of the specific structure of
    Slater determinants, which would be more stable in some cases than the more
    general approach.
    """
    @staticmethod
    def forward(ctx, states, state_indices_collection, x):
        """
            --- INPUT ---

            states: a tuple of many-body states. Each of the state is a tuple of 
                length n containing n orbitals {\phi_j(r)}, each of which is
                represented by a normal function.

            state_indices_collection: a dict. Its key is the index of the tuple
                states, while its value is the number of electron coordinate samples
                at which the slater determinant corresponding to the indexed 
                many-body states (i.e., n orbitals) should be evaluated.
            
            x: the particle coordinates. Generally, x has shape (batch, n, dim).
               For consistency, batch should be equal to sum of all values of the
               dict state_indices_collection.
               Note that x is required to have only one batch dimension.

            --- OUTPUT ---

            The log absolute value of nxn Slater determinants, which has shape (batch,).
        """
        batch, n, _ = x.shape
        D = torch.empty(batch, n, n, device=x.device)
        base_idx = 0
        for idx, times in state_indices_collection.items():
            orbitals = states[idx]
            for i in range(n):
                D[base_idx:base_idx+times, :, i] = orbitals[i](x[base_idx:base_idx+times, ...])
            base_idx += times

        ctx.save_for_backward(x)
        ctx.states, ctx.state_indices_collection = states, state_indices_collection

        _, logabsdet = D.slogdet()
        return logabsdet
    
    @staticmethod
    def backward(ctx, grad_logabsdet):
        x, = ctx.saved_tensors
        states, state_indices_collection = ctx.states, ctx.state_indices_collection
        batch, n, dim = x.shape

        with torch.enable_grad():
            """
                Here in backward, it seems that the Slater matrix has to be created
            again to guarantee the correctness of the implementation, especially for
            higher-order gradients. WHY?
            """
            D = torch.empty(batch, n, n, device=x.device)
            base_idx = 0
            for idx, times in state_indices_collection.items():
                orbitals = states[idx]
                for i in range(n):
                    D[base_idx:base_idx+times, :, i] = orbitals[i](x[base_idx:base_idx+times, ...])
                base_idx += times

            dorbitals = torch.empty(batch, n, dim, n, device=x.device)
            base_idx = 0
            for idx, times in state_indices_collection.items():
                orbitals = states[idx]
                xs = x[base_idx:base_idx+times, ...]
                for i in range(n):
                    orbital_value = orbitals[i](xs)
                    dorbitals[base_idx:base_idx+times, ..., i], = torch.autograd.grad(orbital_value, xs, 
                            grad_outputs=torch.ones_like(orbital_value), create_graph=True)
                base_idx += times

            dlogabsdet = torch.einsum("...ndj,...jn->...nd", dorbitals, D.inverse())
            grad_x = grad_logabsdet[:, None, None] * dlogabsdet
            return None, None, grad_x

def logabsslaterdetmultstates(states, state_indices_collection, x):
    batch, n, _ = x.shape
    D = torch.empty(batch, n, n, device=x.device)
    base_idx = 0
    for idx, times in state_indices_collection.items():
        orbitals = states[idx]
        for i in range(n):
            D[base_idx:base_idx+times, :, i] = orbitals[i](x[base_idx:base_idx+times, ...])
        base_idx += times
    _, logabsdet = D.slogdet()
    return logabsdet
