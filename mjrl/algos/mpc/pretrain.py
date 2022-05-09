
import numpy as np
import torch
import copy

from mjrl.algos.mpc.ensemble_nn_dynamics import EnsembleWorldModel, batch_call


def _unpack_paths(paths):
    s = np.concatenate([p['observations'][:-1] for p in paths])
    a = np.concatenate([p['actions'][:-1] for p in paths])
    sp = np.concatenate([p['observations'][1:] for p in paths])
    return s,a,sp

def _train_models(models, paths, seed=0, checkpoint_freq=1, **job_data):
    num_checkpoints = job_data['fit_epochs'] // checkpoint_freq

    info = dict(dyn_loss=[ [] for _ in range(num_checkpoints)],
                dyn_loss_gen=[ [] for _ in range(num_checkpoints)],
                model_checkpoints=[ [] for _ in range(num_checkpoints)],
                epoch=[ [] for _ in range(num_checkpoints)])

    s, a, sp = _unpack_paths(paths)
    job_data_clone = copy.copy(job_data)
    job_data_clone['fit_epochs'] = 1
    for i, model in enumerate(models):
        print("Training dynamics models {}".format(i))
        for j in range(job_data['fit_epochs']):
            print("Training dynamics models {} epoch {}".format(i, j))
            dynamics_loss = model.fit_dynamics(s, a, sp, **job_data_clone)
            if j % checkpoint_freq==0:
                ind = j//checkpoint_freq
                info['model_checkpoints'][ind].append(copy.deepcopy(model))
                info['dyn_loss'][ind].append(dynamics_loss[-1])
                info['dyn_loss_gen'][ind].append(float(model.compute_loss(s, a, sp)))
                info['epoch'][ind] = j+1

    for i in range(len(info['model_checkpoints'])):
        models = info['model_checkpoints'][i]
        ensemble_model = EnsembleWorldModel(ensemble_size=len(models), state_dim=s.shape[1], act_dim=a.shape[1], seed=seed, **job_data)
        ensemble_model.set_dynamics_from_list(models)
        info['model_checkpoints'][i] = ensemble_model

    return info

def _validate_models(ensemble_model, paths):
    with torch.no_grad():
        s, a, sp = _unpack_paths(paths)
        sp = torch.from_numpy(sp).float().to(device=ensemble_model.device)
        sp_pred = torch.cat(batch_call(ensemble_model.forward, s,a), axis=1)
        error = torch.norm(sp_pred - sp , dim=-1)  # ensemble x data
        delta = torch.cat(batch_call(ensemble_model.compute_delta, s,a))
    return error, delta


def train_dynamics_models(models, paths, slackness=0.1, **job_data):
    """ Train ensemble dynamics model and select early stopping time based on
    uncertainty estimation. """

    # Train multiple models indepdently and compose them as a ensemble model.
    info = _train_models(models, paths, **job_data)

    # Compute errors and predicted errors.
    errors, predicted_errors = [], []
    for ensemble_model in info['model_checkpoints']:
        _errors, _predicted_errors = _validate_models(ensemble_model, paths)
        errors.append(_errors)
        predicted_errors.append(_predicted_errors)
    e = torch.stack(errors)
    pe = torch.stack(predicted_errors).unsqueeze(1)

    # Find the model with that best captures the uncertainty while performing well (wrt slackness).
    score = (((e/pe)<1).view(e.shape[0],-1).float()).mean(axis=1)  # (objective) percentage of correct estimation
    mean_e = e.view(e.shape[0],-1).mean(axis=1)
    ind = torch.where(mean_e < mean_e.min()*(1+slackness))[0]  # (constraint)
    argmax = ind[torch.argmax(score[ind])]  # constrained optimization

    # Extract the corredsponding model.
    ensemble_model = info['model_checkpoints'][argmax]
    info_ = { k:v[argmax]  for k,v in info.items()}
    del info_['model_checkpoints']
    info_['ratio'] = torch.quantile( (e/pe)[argmax], 0.999, dim=1).cpu().numpy()
    info_['score'] = score[ind].max()
    ensemble_model.train_info = info_
    return ensemble_model, info


# def _train_models(models, paths, seed=0, job_data=None):
#     job_data = job_data or {}
#     info = dict(dyn_loss=[], dyn_loss_gen = [])
#     for i, model in enumerate(models):
#         s, a, sp = _unpack_paths(paths)
#         print("Training dynamics models {}".format(i))
#         dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)

#         print('dyn_loss_' + str(i), dynamics_loss[-1])
#         info['dyn_loss'].append(dynamics_loss[-1])
#         loss_general = model.compute_loss(s, a, sp)  # generalization error
#         print('dyn_loss_gen_' + str(i), loss_general)
#         info['dyn_loss_gen'].append(loss_general)
#         # if job_data['learn_reward']:
#         #     reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)
#         #      logger.log_kv('rew_loss_' + str(i), reward_loss[-1])

#       # NOTE Currently, it only supports loading pretrained models.
#     ensemble_model = EnsembleWorldModel(ensemble_size=len(models), state_dim=s.shape[1], act_dim=a.shape[1], seed=seed, **job_data)
#     ensemble_model.set_dynamics_from_list(models)
#     return ensemble_model, info

# def _validate_models(ensemble_model, paths):
#     with torch.no_grad():
#         s, a, sp = _unpack_paths(paths)
#         sp = torch.from_numpy(sp).float().to(device=ensemble_model.device)
#         sp_pred = torch.cat(batch_call(ensemble_model.forward, s,a), axis=1)
#         error = torch.norm(sp_pred - sp , dim=-1)  # ensemble x data
#         delta = torch.cat(batch_call(ensemble_model.compute_delta, s,a))
#     return error, delta



# def train_dynamics_models(models, paths, val_percentage=0.2, job_data=None):

#     # rmax?
#     # paths = paths[:10]
#     paths = copy.copy(paths)
#     paths = np.random.permutation(paths)
#     n_tr_paths = int((1-val_percentage)*len(paths))
#     paths_tr = paths[:n_tr_paths]
#     paths_va = paths[n_tr_paths:]

#     print('Use validation set to tune truncation parameters.')
#     ensemble_model, _ = _train_models(models, paths_tr, job_data=job_data)
#     errors, predicted_errors = _validate_models(ensemble_model, paths_va)
#     ratio, _ = (errors/predicted_errors).max(axis=1)  # over samples
#     print('Training dynamics model on all data.')
#     ensemble_model, info = _train_models(models, paths, job_data=job_data)
#     info['ratio'] = ratio.cpu().numpy()
#     ensemble_model.train_info = info
#     return ensemble_model

##########################################################################################

# def train_dynamics_models(models, paths, val_percentage=0.2, job_data=None):

#     # rmax?
#     models0 = copy.deepcopy(models)
#     paths = copy.copy(paths)
#     # paths = paths[:10]
#     # Select early stopping time
#     print('Use validation set to tune hyperparameters.')
#     paths = np.random.permutation(paths)
#     n_tr_paths = int((1-val_percentage)*len(paths))
#     paths_tr = paths[:n_tr_paths]
#     paths_va = paths[n_tr_paths:]
#     import pickle, os
#     OUT_DIR = './'
#     # info = _train_models(models0, paths_tr, job_data=job_data)
#     # pickle.dump(info, open(os.path.join(OUT_DIR, 'temp_models_info_reg0001.pickle'), 'wb'))

#     # info = pickle.load(open(os.path.join(OUT_DIR, 'temp_models_info_white_output_LeakyReLU.pickle'), 'rb'))
#     # info = pickle.load(open(os.path.join(OUT_DIR, 'temp_models_info.pickle'), 'rb'))
#     info = pickle.load(open(os.path.join(OUT_DIR, 'temp_models_info_white_output.pickle'), 'rb'))


#     np.set_printoptions(precision=3)
#     errors = []
#     predicted_errors = []
#     for ensemble_model in info['model_checkpoints']:
#         _errors, _predicted_errors = _validate_models(ensemble_model, paths_va)
#         errors.append(_errors)
#         predicted_errors.append(_predicted_errors)

#     print('upper bound percentage', np.array([ (((e/pe)<1).sum()/e.numel() ) .cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio max', np.array([ (e/pe).max().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio mean', np.array([ (e/pe).mean().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio min', np.array([ (e/pe).min().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio std', np.array([ (e/pe).std().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))

#     print('error max', np.array([ (e).max().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error mean', np.array([ (e).mean().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error min', np.array([ (e).min().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error std', np.array([ (e).std().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))


#     print('\n')
#     errors = []
#     predicted_errors = []
#     for ensemble_model in info['model_checkpoints']:
#         _errors, _predicted_errors = _validate_models(ensemble_model, paths_tr)
#         errors.append(_errors)
#         predicted_errors.append(_predicted_errors)
#     print('upper bound percentage', np.array([ (((e/pe)<1).sum()/e.numel() ) .cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio max', np.array([ (e/pe).max().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio mean', np.array([ (e/pe).mean().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio min', np.array([ (e/pe).min().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('ratio std', np.array([ (e/pe).std().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))

#     print('error max', np.array([ (e).max().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error mean', np.array([ (e).mean().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error min', np.array([ (e).min().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     print('error std', np.array([ (e).std().cpu().numpy() for e, pe in zip(errors, predicted_errors)]))
#     import pdb; pdb.set_trace()

#     # errors, predicted_errors = _validate_models(ensemble_model, paths_va)
#     # ratio, _ = (errors/predicted_errors).max(axis=1)  # over samples
#     # print('Training dynamics model on all data.')
#     # ensemble_model, info = _train_models(models, paths, job_data=job_data)
#     # info['ratio'] = ratio.cpu().numpy()
#     # ensemble_model.train_info = info
#     # return ensemble_model
