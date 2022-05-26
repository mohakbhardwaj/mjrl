import torch

def scale_ctrl(ctrl, action_lows, action_highs, squash_fn='clamp'):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[None, :, None]
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == 'clamp':
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == 'clamp_rescale':
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == 'tanh':
        ctrl = torch.tanh(ctrl)
    elif squash_fn == 'identity':
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)

def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    cost_seq = torch.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])

    # cost_seq = torch.fliplr(torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq

def cost_to_go_debug(cost_seq, gamma):
    batch_size, horizon = cost_seq.shape
    cost_to_go = torch.zeros(batch_size, device=cost_seq.device)
    for t in range(horizon):
        cost_to_go += (gamma**t) * cost_seq[:,t]
    return cost_to_go
