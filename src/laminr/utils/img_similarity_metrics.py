import torch


def cosine_similarity(tensor, mean_value=None):
    # if mean_value is not None:
    #     tensor = tensor - mean_value
    # else:
    #     tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    tensor_normed = tensor / torch.norm(tensor, p=2, dim=1, keepdim=True)
    return tensor_normed @ tensor_normed.T


def eucl_dist_similarity(x):
    # normalize
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True)

    # make distance
    x = torch.cdist(x, x)
    x = -(x - 1)
    return x


def dot_product_similarity(x):
    x_normed = x / x.norm(dim=1, p=2, keepdim=True)
    return x_normed @ x_normed.T
