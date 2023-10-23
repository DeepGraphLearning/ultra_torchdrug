import torch

from torchdrug.layers import functional


def variadic_topks(input, size, ks, largest=True):
    assert input.dtype == torch.float
    if not input.numel():
        index = torch.zeros(input.shape, dtype=torch.long, device=input.device)
        return input, index

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    abs_max = input[mask].abs().max().item() + 1
    # special case: max = min
    gap = max - min + abs_max * 1e-6
    safe_input = input.clamp(min - gap, max + gap)
    offset = gap * 4
    if largest:
        offset = -offset
    index2sample = torch.repeat_interleave(size)
    values_ext = safe_input + offset * index2sample
    index_ext = values_ext.argsort(dim=0, descending=largest)
    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = index_ext[range]

    return input[index], index


def variadic_sort(input, size, descending=False):
    assert input.dtype == torch.float
    if not input.numel():
        index = torch.zeros(input.shape, dtype=torch.long, device=input.device)
        return input, index

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    abs_max = input[mask].abs().max().item() + 1
    # special case: max = min
    gap = max - min + abs_max * 1e-6
    safe_input = input.clamp(min - gap, max + gap)
    offset = gap * 4
    if descending:
        offset = -offset
    index2sample = torch.repeat_interleave(size)
    input_ext = safe_input + offset * index2sample
    index = input_ext.argsort(dim=0, descending=descending)
    return input[index], index


def variadic_shuffle(input, size):
    rand = torch.rand(size.sum(), device=input.device)
    order = variadic_sort(rand, size)[1]
    return input[order], size


def variadic_unique(input, size, return_inverse=False):
    assert input.dtype == torch.long
    if not input.numel():
        if return_inverse:
            inverse = torch.zeros_like(input)
            return input, size, inverse
        else:
            return input, size

    index2sample = torch.repeat_interleave(size)
    max = input.max().item()
    min = input.min().item()
    # special case: max = min
    offset = max - min + 1

    input_ext = input + offset * index2sample
    if return_inverse:
        input_ext, inverse = input_ext.unique(return_inverse=True)
    else:
        input_ext = input_ext.unique()
    output = (input_ext - min) % offset + min
    index2sample = torch.div(input_ext - min, offset, rounding_mode="floor")
    new_size = bincount(index2sample, minlength=len(size))
    if return_inverse:
        return output, new_size, inverse
    else:
        return output, new_size


def bincount(input, minlength=0):
    # torch.bincount relies on atomic operations
    # which is too slow if there are too many races
    # use torch.bucketize instead
    if not input.numel():
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted = (input.diff() >= 0).all()
    if sorted:
        if minlength == 0:
            minlength = input.max() + 1
        range = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range, input)
        return index.diff()

    return input.bincount(minlength=minlength)
