import numpy as np

def eval_success_robel(paths):
    num_paths = len(paths)
    total_score = 0.0
    for path in paths:
        path_len = path['rewards'].shape[0]
        total_score += (sum(path['env_infos']['score']) / path_len)
    return total_score / num_aths*1.
