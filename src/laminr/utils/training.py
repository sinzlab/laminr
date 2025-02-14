import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def check_activity_requirements(acts, requirements):
    passed = False
    if (
        acts.mean() > requirements["avg"]
        and acts.std() < requirements["std"]
        and acts.min() > requirements["necessary_min"]
    ):
        passed = True
    return passed


class JitteringGridDatamodule:
    def __init__(
        self,
        num_invariances,
        grid_points_per_dim,
        steps_per_epoch,
    ):
        self.num_invariances = num_invariances
        self.grid_points_per_dim = grid_points_per_dim
        self.steps_per_epoch = steps_per_epoch
        self.sigma = 2 * np.pi / self.grid_points_per_dim
        grid = (
            torch.linspace(0, 2 * np.pi, self.grid_points_per_dim + 1)[:-1]
            + self.sigma / 2
        )
        self.grid = torch.stack(
            torch.meshgrid(*[grid for _ in range(self.num_invariances)], indexing='ij'), -1
        ).flatten(0, -2)
        grid0 = torch.linspace(0, 2 * np.pi, self.grid_points_per_dim + 1)[:-1]
        self.grid0 = torch.stack(
            torch.meshgrid(*[grid0 for _ in range(self.num_invariances)], indexing='ij'), -1
        ).flatten(0, -2)
        self.base_grid = self.grid.repeat(self.steps_per_epoch, 1)

    def train_dataloader(self):
        jitter = (
            torch.rand([self.steps_per_epoch, self.num_invariances]) * self.sigma
            - 0.5 * self.sigma
        )
        grids = self.base_grid + jitter.repeat_interleave(
            self.grid_points_per_dim**self.num_invariances, dim=0
        )
        return DataLoader(
            grids,
            batch_size=self.grid_points_per_dim**self.num_invariances,
        )


class ParamReduceOnPlateau:
    def __init__(self, factor=0.8, patience=10, mode="min", threshold=0):
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        if mode == "min":
            self.op = torch.lt
            self.threshold = -threshold
            self.best_metric = torch.tensor(float('inf'))
        if mode == "max":
            self.op = torch.gt
            self.best_metric = torch.tensor(float('-inf'))
        self.num_epochs_no_improvement = 0

    def step(self, metric_to_monitor, value_to_reduce):
        if self.op(metric_to_monitor, self.best_metric + self.threshold):
            self.best_metric = metric_to_monitor
            self.num_epochs_no_improvement = 0
        else:
            self.num_epochs_no_improvement += 1

        if self.num_epochs_no_improvement > self.patience:
            value_to_reduce = value_to_reduce * self.factor
            self.num_epochs_no_improvement = 0
        return value_to_reduce


class ImprovementChecker:
    def __init__(self, patience=5, ignore_diff_smaller_than=1e-3):
        self.current_value = np.nan
        self.iter = 0
        self.has_not_improved_counter = 0
        self.patience = patience
        self.ignore_diff_smaller_than = ignore_diff_smaller_than

    def is_increasing(self, value):
        if self.iter == 0:
            self.current_value = value
            self.iter += 1
        else:
            if value <= self.current_value + self.ignore_diff_smaller_than:
                self.has_not_improved_counter += 1
            else:
                self.has_not_improved_counter = 0
                self.current_value = value
        return False if self.has_not_improved_counter == self.patience else True
