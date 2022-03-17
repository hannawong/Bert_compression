from math import sqrt
from torch.nn import Linear


def copy_same_uniform(W):
    """Return a copy of the tensor intialized at random with the same
    empirical 1st and second moment"""
    out = W.new_empty(W.size())
    a = (sqrt(3) * W.std()).data
    out.uniform_(-a, a) + W.mean()
    return out


def interpolate_linear_layer(layer, mask, dim=-1, other_layer=None):
    """Interpolates between linear layers.
    If the second layer is not provided, interpolate with random"""
    if other_layer is None:
        W = copy_same_uniform(layer.weight)
        if layer.bias is not None:
            b = copy_same_uniform(layer.bias)
    else:
        W = other_layer.weight.clone().detach()
        if layer.bias is not None:
            b = other_layer.bias.clone().detach()
    sizes = [1, 1]
    sizes[dim] = W.size(dim)
    weight_mask = mask.unsqueeze(dim).repeat(*sizes)
    layer.weight.requires_grad = False
    layer.weight.masked_fill_(weight_mask, 0)
    W.masked_fill_(weight_mask.eq(0), 0)
    layer.weight.data += W
    layer.weight.requires_grad = True
    if layer.bias is not None and dim != 0:
        layer.bias.requires_grad = False
        layer.bias.masked_fill_(mask, 0)
        b.masked_fill_(mask.eq(0), 0)
        layer.bias.data += b
        layer.bias.requires_grad = True

def prune_linear_layer(layer, dims, dim=-1):
    """Interpolates between linear layers.
    If the second layer is not provided, interpolate with random"""
    dim = (dim+100) % 2
    W = layer.weight.index_select(dim, dims).clone().detach()
    print(W.size())
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[dims].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(dims)
    new_layer = Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(W.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer



def mask_grad_linear_layer(layer, mask, dim=-1):
    """zeros gradient of certain rows/columns of a linear layer"""
    sizes = [1, 1]
    sizes[dim] = layer.weight.size(dim)
    weight_mask = mask.unsqueeze(dim).repeat(*sizes)
    # Weight
    if layer.weight.grad is not None:
        layer.weight.grad.data.masked_fill_(weight_mask, 0)
        # print(weight_mask)
        # print(layer.weight.grad)
    # Bias
    if layer.bias is not None and layer.bias.grad is not None:
        if dim == 0:
            layer.bias.grad.data.zero_()
        else:
            layer.bias.grad.data.masked_fill_(mask, 0)
