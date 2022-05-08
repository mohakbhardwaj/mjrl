
import numpy as np
import torch
import copy

from mjrl.algos.mpc.ensemble_nn_dynamics import EnsembleWorldModel, batch_call


def _unpack_paths(paths):
    s = np.concatenate([p['observations'][:-1] for p in paths])
    a = np.concatenate([p['actions'][:-1] for p in paths])
    sp = np.concatenate([p['observations'][1:] for p in paths])
    return s,a,sp

def _train_models(models, paths, seed=0, job_data=None):
    job_data = job_data or {}
    info = dict(dyn_loss=[], dyn_loss_gen = [])
    for i, model in enumerate(models):
        s, a, sp = _unpack_paths(paths)
        print("Training dynamics models {}".format(i))
        dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)

        print('dyn_loss_' + str(i), dynamics_loss[-1])
        info['dyn_loss'].append(dynamics_loss[-1])
        loss_general = model.compute_loss(s, a, sp)  # generalization error
        print('dyn_loss_gen_' + str(i), loss_general)
        info['dyn_loss_gen'].append(loss_general)
        # if job_data['learn_reward']:
        #     reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)
        #      logger.log_kv('rew_loss_' + str(i), reward_loss[-1])

      # NOTE Currently, it only supports loading pretrained models.
    ensemble_model = EnsembleWorldModel(ensemble_size=len(models), state_dim=s.shape[1], act_dim=a.shape[1], seed=seed, **job_data)
    ensemble_model.set_dynamics_from_list(models)
    return ensemble_model, info

def _validate_models(ensemble_model, paths):
    with torch.no_grad():
        s, a, sp = _unpack_paths(paths)
        sp = torch.from_numpy(sp).float().to(device=ensemble_model.device)
        sp_pred = torch.cat(batch_call(ensemble_model.forward, s,a), axis=1)
        error = torch.norm(sp_pred - sp , dim=-1)  # ensemble x data
        delta = torch.cat(batch_call(ensemble_model.compute_delta, s,a))
    return error, delta



def train_dynamics_models(models, paths, val_percentage=0.2, job_data=None):

    # rmax?
    # paths = paths[:10]
    paths = copy.copy(paths)
    paths = np.random.permutation(paths)
    n_tr_paths = int((1-val_percentage)*len(paths))
    paths_tr = paths[:n_tr_paths]
    paths_va = paths[n_tr_paths:]

    print('Use validation set to tune truncation parameters.')
    ensemble_model, _ = _train_models(models, paths_tr, job_data=job_data)
    errors, predicted_errors = _validate_models(ensemble_model, paths_va)
    ratio, _ = (errors/predicted_errors).max(axis=1)  # over samples
    print('Training dynamics model on all data.')
    ensemble_model, info = _train_models(models, paths, job_data=job_data)
    info['ratio'] = ratio.cpu().numpy()
    ensemble_model.train_info = info
    return ensemble_model
