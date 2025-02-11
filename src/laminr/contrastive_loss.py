import itertools
import numpy as np
import torch
from torch import nn

from .utils.img_similarity_metrics import cosine_similarity


def get_mask_of_point(
    point,
    grid_points_per_dim,
    num_invariances,
    neighbor_size,
    with_periodic_invariances,
    with_round_neighbor=False,
):
    assert num_invariances == 1 or num_invariances == 2
    ns = int(neighbor_size * grid_points_per_dim)
    if num_invariances == 1:
        assert isinstance(with_periodic_invariances, bool)
        mask = -torch.ones([grid_points_per_dim] * num_invariances)
        x = point[0]
        if x < ns:
            mask[: x + ns + 1] = 1
            if with_periodic_invariances:
                mask[x - ns :] = 1
        elif x + ns >= grid_points_per_dim:
            mask[x - ns :] = 1
            if with_periodic_invariances:
                mask[: ns - grid_points_per_dim + x + 1] = 1
        else:
            mask[x - ns : x + ns + 1] = 1
        mask[x] = 0
        return mask
    if num_invariances == 2:
        point = list(point)
        if isinstance(with_periodic_invariances, bool):
            with_periodic_invariances = [with_periodic_invariances] * 2
        if isinstance(with_periodic_invariances, list):
            assert len(with_periodic_invariances) == 2
        mask_n_points = grid_points_per_dim + 2 * ns
        mask_size = [grid_points_per_dim] * 2
        for dim, periodicity in enumerate(with_periodic_invariances):
            if not periodicity:
                mask_size[dim] = mask_n_points
                point[dim] = point[dim] + ns
        mask = -torch.ones(mask_size)
        start = [int(s / 2) for s in mask_size]
        mask[start[0] - ns : start[0] + ns + 1, start[1] - ns : start[1] + ns + 1] = 1
        mask[start[0], start[1]] = 0
        if with_round_neighbor:
            pos_idxs = np.argwhere(mask > 0)
            pos_idxs = np.array(pos_idxs).T

            for idx in pos_idxs:
                idx = tuple(idx)
                dist_x = idx[0] - start[0]
                dist_y = idx[1] - start[1]
                dist = np.sqrt(dist_x**2 + dist_y**2)
                if dist > ns:
                    mask[idx] = -1
        translation = [p - s for p, s in zip(point, start)]

        mask = np.roll(mask, shift=tuple(translation), axis=(0, 1))

        if not with_periodic_invariances[0]:
            mask = mask[ns : ns + grid_points_per_dim, :]
        if not with_periodic_invariances[1]:
            mask = mask[:, ns : ns + grid_points_per_dim]
        return mask


def GetPosNegMask(
    num_invariances,
    grid_points_per_dim,
    neighbor_size,
    with_periodic_invariances,
    with_round_neighbor,
):
    masks = np.ones([grid_points_per_dim] * 2 * num_invariances)
    masks = np.ones([grid_points_per_dim] * 2 * num_invariances)
    for point in itertools.product(
        np.arange(grid_points_per_dim), repeat=num_invariances
    ):
        masks[tuple(point)] = get_mask_of_point(
            point,
            grid_points_per_dim,
            num_invariances,
            neighbor_size,
            with_periodic_invariances,
            with_round_neighbor,
        )
    return masks


class SimCLROnGrid(nn.Module):
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        neighbor_size,
        temperature,
        with_periodic_invariances,
        with_round_neighbor=False,
        **args,
    ):
        super().__init__()
        self.num_invariances = num_invariances
        self.points_per_dim = grid_points_per_dim
        self.neighbor_size = neighbor_size
        self.temperature = temperature
        self.with_round_neighbor = with_round_neighbor
        self.eps = 1e-6
        self.neighbor_mask = torch.tensor(
            GetPosNegMask(
                num_invariances,
                grid_points_per_dim,
                neighbor_size,
                with_periodic_invariances,
                with_round_neighbor,
            ).astype(np.float32)
        )
        self.register_buffer(
            "flat_neighbor_mask",
            self.neighbor_mask.reshape(
                self.points_per_dim**self.num_invariances,
                self.points_per_dim**self.num_invariances,
            ),
        )
        self.register_buffer(
            "flat_neighbor_mask_pos",
            torch.where(self.flat_neighbor_mask > 0, self.flat_neighbor_mask, 0.0),
        )
        self.register_buffer(
            "flat_neighbor_mask_neg",
            torch.where(self.flat_neighbor_mask < 0, self.flat_neighbor_mask, 0.0) * -1,
        )

    def reg_term(self, images):
        images = images.flatten(
            1, -1
        )  # [N, c, h, w] -> [N, c * h * w]  N=number of points in the grid
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        positive_sim = (similarity * self.flat_neighbor_mask_pos).sum(dim=-1)
        positive_n = self.flat_neighbor_mask_pos.sum(dim=-1)
        # num = positive_sim / positive_n
        num = torch.where(positive_sim == 0, 0.0, positive_sim / positive_n) + self.eps

        negative_sim = (similarity * self.flat_neighbor_mask_neg).sum(dim=-1)
        negative_n = self.flat_neighbor_mask_neg.sum(dim=-1)
        # den = negative_sim / negative_n
        den = torch.where(negative_sim == 0, 0.0, negative_sim / negative_n) + self.eps
        reg_term = torch.log(num / den).mean()
        return reg_term * self.temperature / 2

    def pos_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        num = (similarity * self.flat_neighbor_mask_pos).sum(
            dim=-1
        ) / self.flat_neighbor_mask_pos.sum(dim=-1)
        return torch.log(num).mean() * self.temperature / 2

    def neg_term(self, images):
        images = images.flatten(1, -1)
        similarity = torch.exp(cosine_similarity(images) / self.temperature)
        den = -(-similarity * self.flat_neighbor_mask_neg).sum(
            dim=-1
        ) / self.flat_neighbor_mask_neg.sum(dim=-1)
        return -torch.log(den).mean() * self.temperature / 2
