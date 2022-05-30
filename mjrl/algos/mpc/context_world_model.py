from mjrl.algos.mpc.ensemble_nn_dynamics import EnsembleWorldModel
import numpy as np
import torch

class WorldModelWithContext():
    def __init__(self, state_dim, act_dim, context_dim,
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
                **kwargs):
    
        self.context_dim = context_dim
        self.world_model = EnsembleWorldModel(
            state_dim, 
            act_dim,
            ensemble_size,
            learn_reward,
            hidden_size,
            seed,
            fit_lr,
            fit_wd,
            device,
            activation,
            residual,
            *args,
            **kwargs)

    def forward(self, s, a):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        if len(s.shape) == 2:
            s = s.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x dim
            a = a.repeat(self.ensemble_size, 1, 1)  # ensemble x batch x dim

        ctxt_vec = s[:,:,-self.context_dim:]
        s_minus_ctxt = s[:,:,0:-self.context_dim]
        s_next = self.world_model.forward(s_minus_ctxt, a)
        s_next = torch.cat([s_next, ctxt_vec], dim=-1)
        return s_next

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        ctxt_vec = s[:,:,-self.context_dim:]
        s_minus_ctxt = s[:,:,0:-self.context_dim]
        s_next = self.world_model.forward(s_minus_ctxt, a)
        s_next = torch.cat([s_next, ctxt_vec], dim=-1)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def fit_dynamics(slef, *args, **kwargs):
        raise NotImplementedError

    def set_dynamics_from_list(self, list_of_WMs):
        self.world_model.set_dynamics_from_list(list_of_WMs)

    def compute_delta(self, *args):  # this is a numpy method
        B = args[0].shape[0]  # batch
        preds = self.forward(*args) # Bxd, Bxd -> ExBxd
        delta = torch.zeros(B, device=self.device)
        for i in range(self.ensemble_size):
            for j in range(i+1,self.ensemble_size):
                delta = torch.maximum(delta, torch.norm(preds[i]-preds[j], dim=1))
        return delta

    @property
    def device(self):
        return self.world_model.device

    @property
    def ensemble_size(self):
        return self.world_model.ensemble_size

    def __len__(self):
        return self.world_model.ensemble_size
