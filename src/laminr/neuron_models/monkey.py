import warnings
import numpy as np
import torch
import torch.nn as nn
from laminr.neuron_models.monkey_model_utils import (
    se_core_full_gauss_readout,
    se_core_point_readout,
)
from neuralpredictors.layers.readouts import PointPooled2d, FullGaussian2d
from huggingface_hub import hf_hub_download

data_info = {
    "3631896544452": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 32,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3632669014376": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 21,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3632932714885": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 11,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3633364677437": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634055946316": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 21,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634142311627": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634658447291": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 5,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3634744023164": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 12,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3635178040531": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 6,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3635949043110": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3636034866307": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 20,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3636552742293": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 24,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637161140869": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 22,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637248451650": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 7,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637333931598": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 9,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637760318484": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 20,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3637851724731": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638367026975": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 16,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638456653849": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 2,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638885582960": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 5,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638373332053": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 10,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638541006102": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 17,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638802601378": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 7,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3638973674012": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 18,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639060843972": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 12,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639406161189": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 14,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3640011636703": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 2,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639664527524": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 18,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639492658943": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 17,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3639749909659": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 8,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3640095265572": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 25,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
    "3631807112901": {
        "input_dimensions": torch.Size([128, 1, 93, 93]),
        "input_channels": 1,
        "output_dimension": 29,
        "img_mean": 114.54466,
        "img_std": 64.07781,
    },
}

pointpool_model_config = {
    "pad_input": False,
    "stack": -1,
    "depth_separable": True,
    "input_kern": 24,
    "gamma_input": 10,
    "gamma_readout": 0.5,
    "hidden_dilation": 2,
    "hidden_kern": 9,
    "se_reduction": 16,
    "n_se_blocks": 2,
    "hidden_channels": 32,
    "linear": False,
}

gaussian_model_config = {
    "pad_input": False,
    "stack": -1,
    "depth_separable": True,
    "input_kern": 24,
    "gamma_input": 10,
    "gamma_readout": 0.5,
    "hidden_dilation": 2,
    "hidden_kern": 9,
    "se_reduction": 16,
    "hidden_channels": 32,
    "linear": False,
    "n_se_blocks": 0,
}


def get_pointpool_monkey_model(
    seed=None, data_info=data_info, model_config=pointpool_model_config
):
    seed = seed if seed is not None else np.random.randint(0, 100)
    model = se_core_point_readout(seed=seed, data_info=data_info, **model_config)
    return model


class keyless_pointpool_monkey_model(nn.Module):
    def __init__(self, model_initial):
        super().__init__()
        model_initial.eval()
        keys = list(model_initial.readout.keys())
        att = dict(**model_initial.readout[keys[0]].__dict__)
        att["bias"] = True
        att["outdims"] = 458
        att["pool_steps"] = 2
        start = 0
        readout = PointPooled2d(**att)
        for k in keys:
            r = model_initial.readout[k]
            add = r.outdims
            end = start + add
            readout.features.data[:, :, :, start:end] = r.features.data
            readout.grid.data[:, start:end, :, :] = r.grid.data
            readout.bias.data[start:end] = r.bias.data
            start = start + add
        self.core = model_initial.core
        self.readout = readout
        self.eval()

    def forward(self, x):
        x = self.readout(self.core(x))
        return torch.nn.functional.elu(x) + 1


class ensamble_keyless_pointpool_monkey_model(nn.Module):
    def __init__(self, models_paths):
        super().__init__()
        models = []
        for path in models_paths:
            model = get_pointpool_monkey_model()
            model.load_state_dict(torch.load(path))
            models.append(model)
        self.models = nn.ModuleList(
            keyless_pointpool_monkey_model(model) for model in models
        )
        self.eval()

    def forward(self, x):
        x = torch.stack([model(x) for model in self.models]).mean(dim=0)
        return x


def get_gaussian_monkey_model(
    seed=None, data_info=data_info, model_config=gaussian_model_config
):
    seed = seed if seed is not None else np.random.randint(0, 100)
    model = se_core_full_gauss_readout(seed=seed, data_info=data_info, **model_config)
    return model


class keyless_gaussian_monkey_model(nn.Module):
    def __init__(self, model_initial):
        super().__init__()
        model_initial.eval()
        keys = list(model_initial.readout.keys())
        att = dict(**model_initial.readout[keys[0]].__dict__)
        att["bias"] = True
        att["outdims"] = 458
        start = 0
        readout = FullGaussian2d(**att)
        for k in keys:
            r = model_initial.readout[k]
            add = r.outdims
            end = start + add
            readout.features.data[:, :, :, start:end] = r.features.data
            readout.grid.data[:, start:end, :, :] = r.grid.data
            readout.bias.data[start:end] = r.bias.data
            start = start + add
        self.core = model_initial.core
        self.readout = readout
        self.eval()

    def forward(self, x):
        x = self.readout(self.core(x))
        return torch.nn.functional.elu(x) + 1


class ensamble_keyless_gaussian_monkey_model(nn.Module):
    def __init__(self, models_paths):
        super().__init__()
        models = []
        for path in models_paths:
            model = get_gaussian_monkey_model()
            model.load_state_dict(torch.load(path))
            models.append(model)
        self.models = nn.ModuleList(
            keyless_gaussian_monkey_model(model) for model in models
        )
        self.eval()

    def forward(self, x):
        x = torch.stack([model(x) for model in self.models]).mean(dim=0)
        return x


def monkey_v1(model_type="pointpool"):
    if model_type == "pointpool":
        # download model weights
        model_paths = [
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_pointpool_1.pt",
            ),
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_pointpool_2.pt",
            ),
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_pointpool_3.pt",
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return ensamble_keyless_pointpool_monkey_model(model_paths)
    elif model_type == "gaussian":
        # download model weights
        model_paths = [
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_gaussian_1.pt",
            ),
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_gaussian_2.pt",
            ),
            hf_hub_download(
                repo_id="mobashiri/bashiri_baroni_iclr2025",
                filename=f"monkey_gaussian_3.pt",
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return ensamble_keyless_gaussian_monkey_model(model_paths)
    else:
        raise ValueError("readout_type must be 'pointpool' or 'gaussian'")
