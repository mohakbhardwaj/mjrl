import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
# from mjrl.utils.optimize_model import fit_data
from mjrl.utils.tensor_utils import tensorize
import math
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from mjrl.algos.mpc.ensemble_nn_dynamics import batch_call
from tqdm import tqdm

import pickle

class ValueFunctionNet(nn.Module):
    def __init__(self, state_dim, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, device='cpu', hidden_size=(128, 128), *args, **kwargs):
        
        super().__init__()
        self.state_dim, self.hidden_size = state_dim, hidden_size
        self.out_dim = 1
        self.layer_sizes = (state_dim, ) + hidden_size + (self.out_dim, )

        # self.n = inp_dim if inp_dim is not None else env_spec.observation_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.device = device
        self.inp = inp
        self.hidden_size = hidden_size

        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, s):
        out = s
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        # transforms = (self.s_shift, self.s_scale,
        #               self.a_shift, self.a_scale,
        #               self.out_shift, self.out_scale)
        # transforms = self.transformations
        return dict(weights=network_weights) #, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        # s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        # self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)
        # self.set_transformations(**new_params['transforms'])

    def fit(self, paths, return_errors=False):

        s = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths])

        assert s.shape[0] == returns.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            returns = torch.from_numpy(returns).float()
        s = s.to(self.device); returns = returns.to(self.device)

        if return_errors:
            with torch.no_grad():
                predictions = torch.cat(batch_call(self.forward, s)).cpu().numpy().ravel()
                returns_np = returns.cpu().numpy().ravel()
                errors =  returns_np - predictions
                error_before = np.sum(errors**2)/(np.sum(returns_np**2) + 1e-8).item()
        
        X = (s,); Y = returns.unsqueeze(-1) #/ (out_scale + 1e-8)

        epoch_losses = self.fit_model(X, Y, self.optimizer,
                            self.loss_function, self.batch_size, self.epochs)

        if return_errors:
            with torch.no_grad():
                predictions = torch.cat(batch_call(self.forward, s)).cpu().numpy().ravel()# .to('cpu').data.numpy().ravel()
                errors = returns_np - predictions #.ravel()
                error_after = np.sum(errors**2)/(np.sum(returns_np**2) + 1e-8).item()
                return error_before, error_after, epoch_losses


    def fit_model(self, X, Y, optimizer, loss_func,
                batch_size, epochs, max_steps=1e10):
        """
        :param nn_model:        pytorch model of form Y = f(*X) (class)
        :param X:               tuple of necessary inputs to the function
        :param Y:               desired output from the function (tensor)
        :param optimizer:       optimizer to use
        :param loss_func:       loss criterion
        :param batch_size:      mini-batch size
        :param epochs:          number of epochs
        :return:
        """
        assert type(X) == tuple
        for d in X: assert type(d) == torch.Tensor
        assert type(Y) == torch.Tensor
        device = Y.device
        for d in X: assert d.device == device

        num_samples = Y.shape[0]
        epoch_losses = []
        steps_so_far = 0
        for ep in tqdm(range(epochs)):
            rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
            ep_loss = 0.0
            num_steps = int(num_samples // batch_size)
            for mb in range(num_steps):
                data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
                batch_X  = [d[data_idx] for d in X]
                batch_Y  = Y[data_idx]
                optimizer.zero_grad()
                Y_hat    = self.forward(*batch_X)
                loss = loss_func(Y_hat, batch_Y)
                loss.backward()
                optimizer.step()
                ep_loss += loss.to('cpu').data.numpy()
            epoch_losses.append(ep_loss * 1.0/num_steps)
            steps_so_far += num_steps
            if steps_so_far >= max_steps:
                print("Number of grad steps exceeded threshold. Terminating early..")
                break
        return epoch_losses




# import numpy as np
# import copy
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# # from mjrl.utils.optimize_model import fit_data
# from mjrl.utils.tensor_utils import tensorize
# import math
# from torch import Tensor
# from torch.nn.parameter import Parameter, UninitializedParameter
# from mjrl.algos.mpc.ensemble_nn_dynamics import batch_call, EnsembleLinear
# from tqdm import tqdm

# import pickle

# class ValueFunctionNet(nn.Module):
#     def __init__(self, state_dim, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
#                  batch_size=64, epochs=1, device='cpu', hidden_size=(128, 128), ensemble_size=1, *args, **kwargs):
        
#         super().__init__()
#         self.state_dim, self.hidden_size, self.ensemble_size = state_dim, hidden_size, ensemble_size
#         self.out_dim = 1
#         self.layer_sizes = (state_dim, ) + hidden_size + (self.out_dim, )

#         # self.n = inp_dim if inp_dim is not None else env_spec.observation_dim
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.reg_coef = reg_coef
#         self.device = device
#         self.inp = inp

#         # self.fc_layers = nn.ModuleList([EnsembleLinear(self.layer_sizes[i], self.layer_sizes[i+1], self.ensemble_size)
#         #                                 for i in range(len(self.layer_sizes)-1)])
#         self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
#                                         for i in range(len(self.layer_sizes)-1)])
#         self.nonlinearity = torch.relu

#         self.to(self.device)

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=reg_coef)
#         self.loss_function = torch.nn.MSELoss()

#     def forward(self, s):
#         # if len(s.shape) == 2:
#         #     s = s.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x dim
#         out = s
#         for i in range(len(self.fc_layers)-1):
#             out = self.fc_layers[i](out)
#             out = self.nonlinearity(out)
#         out = self.fc_layers[-1](out)
#         return out

#     def get_params(self):
#         network_weights = [p.data for p in self.parameters()]
#         # transforms = (self.s_shift, self.s_scale,
#         #               self.a_shift, self.a_scale,
#         #               self.out_shift, self.out_scale)
#         # transforms = self.transformations
#         return dict(weights=network_weights) #, transforms=transforms)

#     def set_params(self, new_params):
#         new_weights = new_params['weights']
#         # s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = new_params['transforms']
#         for idx, p in enumerate(self.parameters()):
#             p.data = new_weights[idx]
#         # self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)
#         # self.set_transformations(**new_params['transforms'])

#     def fit(self, paths, return_errors=False):

#         s = np.concatenate([p['observations'] for p in paths])
#         returns = np.concatenate([p['returns'] for p in paths])

#         assert s.shape[0] == returns.shape[0]
#         if type(s) == np.ndarray:
#             s = torch.from_numpy(s).float()
#             returns = torch.from_numpy(returns).float()
#         s = s.to(self.device); returns = returns.to(self.device)

#         if return_errors:
#             with torch.no_grad():
#                 predictions = torch.cat(batch_call(self.forward, s)).cpu().numpy().ravel()
#                 returns_np = returns.cpu().numpy().ravel()
#                 errors =  returns_np - predictions
#                 error_before = np.sum(errors**2)/(np.sum(returns_np**2) + 1e-8).item()
        
#         X = (s,); Y = returns.unsqueeze(-1) #/ (out_scale + 1e-8)

#         epoch_losses = self.fit_model(X, Y, self.optimizer,
#                             self.loss_function, self.batch_size, self.epochs)

#         if return_errors:
#             with torch.no_grad():
#                 predictions = torch.cat(batch_call(self.forward, s)).cpu().numpy().ravel()# .to('cpu').data.numpy().ravel()
#                 errors = returns_np - predictions #.ravel()
#                 error_after = np.sum(errors**2)/(np.sum(returns_np**2) + 1e-8).item()
#                 return error_before, error_after, epoch_losses


#     def fit_model(self, X, Y, optimizer, loss_func,
#                 batch_size, epochs, ensemble_size=1, max_steps=1e10):
#         """
#         :param nn_model:        pytorch model of form Y = f(*X) (class)
#         :param X:               tuple of necessary inputs to the function
#         :param Y:               desired output from the function (tensor)
#         :param optimizer:       optimizer to use
#         :param loss_func:       loss criterion
#         :param batch_size:      mini-batch size
#         :param ensemble_size:   number of ensemble members
#         :param epochs:          number of epochs
#         :return:
#         """
#         assert type(X) == tuple
#         for d in X: assert type(d) == torch.Tensor
#         assert type(Y) == torch.Tensor
#         device = Y.device
#         for d in X: assert d.device == device

#         num_samples = Y.shape[0]
#         epoch_losses = []
#         steps_so_far = 0
#         for ep in tqdm(range(epochs)):
#             rand_idx = torch.LongTensor(np.random.permutation(num_samples)).unsqueeze(0).to(device)
#             # rand_idx = torch.cat(rand_idx, dim=0)
#             ep_loss = 0.0
#             num_steps = int(num_samples // batch_size)
#             for mb in range(num_steps):
#                 data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
#                 batch_X  = [d[data_idx] for d in X]
#                 batch_Y  = Y[data_idx]
#                 optimizer.zero_grad()
#                 Y_hat    = self.forward(*batch_X)
#                 loss = loss_func(Y_hat, batch_Y)
#                 loss.backward()
#                 optimizer.step()
#                 ep_loss += loss.to('cpu').data.numpy()
#             epoch_losses.append(ep_loss * 1.0/num_steps)
#             steps_so_far += num_steps
#             if steps_so_far >= max_steps:
#                 print("Number of grad steps exceeded threshold. Terminating early..")
#                 break
#         return epoch_losses
