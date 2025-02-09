from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class LinearPerDim(nn.Module):
    def __init__(self, n_dim, in_dim, out_dim, bias=True):
        super().__init__()
        sqrt_k = np.sqrt(1 / in_dim)
        self.weight = nn.Parameter(torch.rand(n_dim, in_dim, out_dim) * 2 * sqrt_k - sqrt_k)
        self.has_bias = False
        if bias:
            self.bias = nn.Parameter(torch.rand(n_dim, 1, out_dim) * 2 * sqrt_k - sqrt_k)
            self.has_bias = True
        
    def forward(self, x):
        out = torch.einsum("dni,dio->dno", x, self.weight)
        return out + self.bias if self.has_bias else out

class CouplingLayer(nn.Module):
    def __init__(self, n_dim, in_dim, untouched_half, affine=False, hidden_features=256, hidden_layers=3, init_identity=True, eps=1e-8):
        super().__init__()
        self.n_dim = n_dim
        self.in_dim = in_dim
        self.untouched_half = untouched_half
        self.affine = affine
        self.init_identity = init_identity
        self.eps = eps
        self.fhi, self.shi = self.dim_indices()
        
        in_dim = len(self.fhi) if self.untouched_half == 'first' else len(self.shi)
        out_dim = len(self.shi) if self.untouched_half == 'first' else len(self.fhi)
        self.t = TranslationNet(n_dim, in_dim, out_dim, hidden_features=hidden_features, hidden_layers=hidden_layers)
        if init_identity:
            self.t.net[-1].weight.data.zero_()
            self.t.net[-1].bias.data.zero_()
        if affine:
            if init_identity:
                self.s = ScaleNet(n_dim, in_dim, out_dim, hidden_features=hidden_features, hidden_layers=hidden_layers)
                self.s.net[-1].weight.data.zero_()
                self.s.net[-1].bias.data.zero_()
        
    def dim_indices(self):
        indices = np.arange(self.in_dim)
        first_half = indices[:self.in_dim//2]
        second_half = indices[self.in_dim//2:]
        return first_half, second_half

    def forward(self, x):
        x1 = x[..., self.fhi]
        x2 = x[..., self.shi]
        
        if self.untouched_half == 'first':
            z1 = x1
            positive_scale = torch.exp(self.s(x1)) if self.affine else 1. 
#             positive_scale = F.elu(self.s(x1)) + 1 + self.eps if self.affine else 1. 
            z2 = x2 * positive_scale + self.t(x1)
        elif self.untouched_half == 'second':
            positive_scale = torch.exp(self.s(x2)) if self.affine else 1.
#             positive_scale = F.elu(self.s(x2)) + 1 + self.eps if self.affine else 1.
            z1 = x1 * positive_scale + self.t(x2)
            z2 = x2
        
        if (len(z1) == 1) and (len(z2) != 1):
            _z1 = z1.repeat(len(z2), 1, 1)
            return torch.cat((_z1, z2), dim=-1)
        elif len(z2) == 1 and (len(z1) != 1):
            _z2 = z2.repeat(len(z1), 1, 1)
            return torch.cat((z1, _z2), dim=-1)
        else:
            return torch.cat((z1, z2), dim=-1)
    
    def inverse(self, z):
        z1 = z[..., self.fhi]
        z2 = z[..., self.shi]
        
        if self.untouched_half == 'first':
            x1 = z1
            positive_scale = torch.exp(self.s(z1)) if self.affine else 1.
            # positive_scale = F.elu(self.s(z1)) + 1 + self.eps if self.affine else 1.
            x2 = (z2 - self.t(z1)) / positive_scale
        elif self.untouched_half == 'second':
            positive_scale = torch.exp(self.s(z2)) if self.affine else 1.
            # positive_scale = F.elu(self.s(z2)) + 1 + self.eps if self.affine else 1.
            x1 = (z1 - self.t(z2)) / positive_scale
            x2 = z2
            
        return torch.cat((x1, x2), dim=-1)
    

class ScaleNet(nn.Module):
    def __init__(self, n_dim, in_dim, out_dim, hidden_features=256, hidden_layers=3):
        super().__init__()
        
        if hidden_layers == 0:
            hidden_features = out_dim
    
        layers = [
            LinearPerDim(
                n_dim,
                in_dim,
                hidden_features,
            )
        ]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ReLU(),
                    LinearPerDim(
                        n_dim, 
                        hidden_features,
                        hidden_features
                        if i < hidden_layers - 1
                        else out_dim,
                    ),
                ]
            )   
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TranslationNet(nn.Module):
    def __init__(self, n_dim, in_dim, out_dim, hidden_features=256, hidden_layers=3):
        super().__init__()
        
        if hidden_layers == 0:
            hidden_features = out_dim
        
        layers = [
            LinearPerDim(
                n_dim,
                in_dim,
                hidden_features,
            )
        ]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ReLU(),
                    LinearPerDim(
                        n_dim,
                        hidden_features,
                        hidden_features
                        if i < hidden_layers - 1
                        else out_dim,
                    ),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RealNVP(nn.Module):
    def __init__(self, n_dim, n_layers, in_dim, affine=False, init_identity=True, hidden_features=256, hidden_layers=3):
        super().__init__()
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.layers = nn.ModuleList([
            CouplingLayer(
                n_dim=n_dim,
                in_dim=in_dim, 
                untouched_half=['first', 'second'][i%in_dim], 
                affine=affine,
                init_identity=init_identity,
                hidden_features=hidden_features, 
                hidden_layers=hidden_layers) for i in range(n_layers)
        ])
    
    def inverse(self, z):
        x = z
        for l in self.layers[::-1]:
            x = l.inverse(x)
        
        return x
    
    def forward(self, x):
        z = x
        for l in self.layers:
            z = l.forward(z)

        return z
    
    
class AffineTransformOld(nn.Module):
    def __init__(self, n_neurons, in_dim=2, stochastic=False, init_noise_scale=.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.in_dim = in_dim
        self.stochastic = stochastic
        self.As = nn.Parameter(torch.stack([torch.eye(in_dim)]*n_neurons))
        self.translation = nn.Parameter(torch.zeros(n_neurons, 1, in_dim))

        if stochastic:
            self.init_noise_scale = init_noise_scale
            self._As_noise_scale = nn.Parameter(torch.log(torch.ones(n_neurons, in_dim, in_dim) * init_noise_scale))
            self._translation_noise_scale = nn.Parameter(torch.log(torch.ones(n_neurons, 1, in_dim) * init_noise_scale))
    
    @property
    def As_noise_scale(self):
        return torch.exp(self._As_noise_scale)

    @property
    def translation_noise_scale(self):
        return torch.exp(self._translation_noise_scale)
        
    def forward(self, x, stochastic=None):
        stochastic = stochastic if stochastic is not None else self.stochastic
        if not self.training:
            stochastic = False
        if stochastic:
            As = self.As + torch.randn_like(self.As) * self.As_noise_scale
            translation = self.translation + torch.randn_like(self.translation) * self.translation_noise_scale
        else:
            As = self.As
            translation = self.translation
        output = torch.einsum('nab, ncb -> nca', As, x) + translation
        return  output


class AffineTransform(nn.Module):
    def __init__(
        self, 
        n_neurons, 
        in_dim=2,
        stochastic=False,
        init_noise_scale=.1,
        allow_scale=True,
        allow_shear=True,
        uniform_scale=False
        ):
        super().__init__()
        self.n_neurons = n_neurons
        self.in_dim = in_dim
        self.stochastic = stochastic
        self.uniform_scale = uniform_scale
        # Parameterize the rotation as angles
        self.angles = nn.Parameter(torch.zeros(n_neurons))  # Rotation angles for each neuron
        
        if allow_scale:
            # Learnable diagonal elements of the scaling matrix
            if uniform_scale: 
                self.scalings = nn.Parameter(torch.ones(n_neurons, 1)) # Only one diagonal elements is learnable
            else:
                self.scalings = nn.Parameter(torch.ones(n_neurons, in_dim)) # Only the diagonal elements are learnable
        else:
            self.register_buffer('scalings', torch.ones(n_neurons, in_dim))

        if allow_shear:
            # Learnable off-diagonal shear elements
            self.shears = nn.Parameter(torch.zeros(n_neurons, in_dim))  # Only learn the shear terms
        else:
            self.register_buffer('shears', torch.zeros(n_neurons, in_dim))

        self.translation = nn.Parameter(torch.zeros(n_neurons, 1, in_dim))

        if stochastic:
            self.init_noise_scale = init_noise_scale
            self._As_noise_scale = nn.Parameter(torch.log(torch.ones(n_neurons, in_dim, in_dim) * init_noise_scale))
            self._translation_noise_scale = nn.Parameter(torch.log(torch.ones(n_neurons, 1, in_dim) * init_noise_scale))
            
    @property
    def As_noise_scale(self):
        return torch.exp(self._As_noise_scale)

    @property
    def translation_noise_scale(self):
        return torch.exp(self._translation_noise_scale)
    
    @property
    def rotation_matrices(self):
        cos_theta = torch.cos(self.angles)
        sin_theta = torch.sin(self.angles)
        return torch.stack([torch.stack([cos_theta, -sin_theta], dim=-1),
                            torch.stack([sin_theta, cos_theta], dim=-1)], dim=-2)
    
    @property
    def scale_matrices(self):
        if self.uniform_scale:
            return torch.diag_embed(self.scalings.repeat(1, 2))
        else:  
            return torch.diag_embed(self.scalings)
    
    @property
    def shear_matrices(self):
        return torch.stack([torch.stack([torch.ones_like(self.shears[:, 0]), self.shears[:, 0]], dim=-1),
                            torch.stack([self.shears[:, 1], torch.ones_like(self.shears[:, 1])], dim=-1)], dim=-2)
    
    @property
    def As(self):
        return torch.einsum('nab,nbc,ncd -> nad', self.rotation_matrices, self.scale_matrices, self.shear_matrices)
        
    def forward(self, x, stochastic=None):
        stochastic = stochastic if stochastic is not None else self.stochastic
        if not self.training:
            stochastic = False
        if stochastic:
            As = self.As + torch.randn_like(self.As) * self.As_noise_scale
            translation = self.translation + torch.randn_like(self.translation) * self.translation_noise_scale
        else:
            As = self.As
            translation = self.translation
        output = torch.einsum('nab, ncb -> nca', As, x) + translation
        return  output



class CoordinateTransform(nn.Module):
    def __init__(
        self,
        n_neurons,
        only_affine=False,
        flow_kwgs=None,
        in_dim=2,
        stochastic=False,
        init_noise_scale=.1,
        allow_scale=True,
        allow_shear=True, 
        uniform_scale=False
        ):
        super().__init__()
        self.n_dim = n_neurons
        self.only_affine = only_affine
        self.stochastic = stochastic
        self.init_noise_scale = init_noise_scale
        self.allow_scale = allow_scale
        self.allow_shear = allow_shear
        self.uniform_scale = uniform_scale
        transforms = []
        if not only_affine:
            if flow_kwgs is None:
                flow_kwgs = dict(n_layers=2, in_dim=2, affine=True, init_identity=True, hidden_features=16, hidden_layers=1)
            else:
                flow_kwgs["in_dim"] = in_dim
            transforms.append(("Warp", RealNVP(n_neurons, **flow_kwgs)))
        transforms.append(("Affine", AffineTransform(
            n_neurons, 
            in_dim=in_dim, 
            stochastic=stochastic, 
            init_noise_scale=init_noise_scale,             
            allow_scale=allow_scale,
            allow_shear=allow_shear,
            uniform_scale=uniform_scale)))
        self.transforms = nn.Sequential(OrderedDict(transforms))
        
    def forward(self, coords):
        if len(coords.shape) == 2:
            coords.unsqueeze_(0)
        return self.transforms(coords)