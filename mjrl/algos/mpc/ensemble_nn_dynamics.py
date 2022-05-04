from mjrl.algos.mbrl.nn_dynamics import WorldModel, DynamicsNet, RewardNet
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter


BATCH_SIZE=10000
import numpy as np

def batch_call(method, *args):
    N = args[0].shape[0]
    outputs = []
    for i in range(N // BATCH_SIZE +1):
        ind_st = i*BATCH_SIZE
        ind_ed = min((i+1)*BATCH_SIZE,N)
        outputs.append(method(*[d[ind_st:ind_ed] for d in args ]))
    return outputs


class EnsembleLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, ensemble_size : int = 2, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weights = Parameter(torch.empty((ensemble_size, out_features, in_features), **factory_kwargs))
        if bias:
            self.biases = Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for w in self.weights:
            # w.transpose_(0, 1)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            # w.transpose_(0, 1)

        if self.biases is not None:
            for w, b in zip(self.weights, self.biases):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2: # batch x dim
            input = input.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x in_features
        return torch.baddbmm(self.biases.unsqueeze(1), input, self.weights.transpose(1,2))  # ensemble x batch x out_features

    def extra_repr(self) -> str:
        return 'ensemble_size = {}, in_features={}, out_features={}, biases={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.biases is not None
        )


class EnsembleWorldModel(WorldModel):
    def __init__(self, state_dim, act_dim,
                 ensemble_size=2,
                 learn_reward=False,
                 hidden_size=(64,64),
                 seed=123,
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu',
                 residual=True,
                 *args,
                 **kwargs,):

        self.ensemble_size = ensemble_size
        super().__init__(state_dim, act_dim, learn_reward=learn_reward, hidden_size=hidden_size, seed=seed, fit_lr=fit_lr, fit_wd=fit_wd, device=device, activation=activation, *args, **kwargs)
        self.dynamics_net = EnsembleDynamicsNet(state_dim, act_dim, hidden_size=hidden_size, ensemble_size=ensemble_size, residual=residual, seed=seed).to(self.device)
        self.dynamics_net.set_transformations()  # in case device is different from default, it will set transforms correctly
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)

    def fit_dynamics(slef, *args, **kwargs):
        raise NotImplementedError()

    def set_dynamics_from_list(self, list_of_WMs):
        self.dynamics_net.set_params_from_list([ m.dynamics_net.get_params() for m in list_of_WMs])

    def compute_delta(self, *args):  # this is a numpy method
        B = args[0].shape[0]  # batch
        preds = self.forward(*args) # Bxd, Bxd -> ExBxd
        delta = torch.zeros(B, device=self.device)
        for i in range(self.ensemble_size):
            for j in range(i+1,self.ensemble_size):
                delta = torch.maximum(delta, torch.norm(preds[i]-preds[j], dim=1))
        return delta

    def __len__(self):
        return self.ensemble_size

class EnsembleDynamicsNet(DynamicsNet):

    def __init__(self, *args, ensemble_size=2, **kwargs):
        self.ensemble_size = ensemble_size
        super().__init__(*args, **kwargs)
        self.fc_layers = nn.ModuleList([EnsembleLinear(self.layer_sizes[i], self.layer_sizes[i+1], ensemble_size)
                                        for i in range(len(self.layer_sizes)-1)])

    def set_transformations(self, *args, **kwargs):
        super().set_transformations(*args, **kwargs)
        for k, v in self.transformations.items():
            if len(v.shape)==1:  # dim
                self.transformations[k] = torch.unsqueeze(v.repeat(self.ensemble_size,1),1)  # ensemble x batch x dim
            if len(v.shape)==2:  # ensemble x dim
                self.transformations[k] = torch.unsqueeze(v,1)  # ensemble x batch x dim
            assert len(self.transformations[k].shape)==3

        for k in self.transformations:
            exec('self.'+k+'=self.transformations[k]')
        self.mask = self.out_scale >= 1e-8


    def forward(self, s, a):
        if len(s.shape) == 2:
            s = s.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x dim
            a = a.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x dim
        return super().forward(s, a)

    def set_params_from_list(self, new_params_list):
        for idx, p in enumerate(self.parameters()):
            new_data = torch.stack([ new_params['weights'][idx] for new_params in  new_params_list   ] )
            assert new_data.shape==p.data.shape, [new_data.shape, p.data.shape]
            p.data = new_data
        for k in self.transformations:
            self.transformations[k] =  torch.stack([new_params['transforms'][k] for new_params in  new_params_list ])
        self.set_transformations(**self.transformations)

    def __len__(self):
        return self.ensemble_size