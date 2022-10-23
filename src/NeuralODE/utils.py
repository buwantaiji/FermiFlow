import torch
torch.set_default_dtype(torch.float64)

def flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])

def shapes_numels(tensors):
    return tuple((tensor.shape, tensor.numel()) for tensor in tensors)

def unflatten(tensor, shapes_numels):
    baseidx = 0
    unflatten_tensors = []
    for shape, numel in shapes_numels:
        unflatten_tensors.append(tensor[baseidx:(baseidx+numel)].reshape(shape))
        baseidx += numel
    return tuple(unflatten_tensors)

def require_grad(tensors, flag):
    return tuple(tensor.requires_grad_(flag) for tensor in tensors)
