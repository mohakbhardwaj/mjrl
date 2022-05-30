import numpy as np
import torch

OBS_ELEMENT_INDICES = {
    'bottom burner': torch.tensor([11, 12]),
    'top burner': torch.tensor([15, 16]),
    'light switch': torch.tensor([17, 18]),
    'slide cabinet': torch.tensor([19]),
    'hinge cabinet': torch.tensor([20, 21]),
    'microwave': torch.tensor([22]),
    'kettle': torch.tensor([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    'bottom burner': torch.tensor([-0.88, -0.01]),
    'top burner': torch.tensor([-0.92, -0.01]),
    'light switch': torch.tensor([-0.69, -0.05]),
    'slide cabinet': torch.tensor([0.37]),
    'hinge cabinet': torch.tensor([0., 1.45]),
    'microwave': torch.tensor([-0.75]),
    'kettle': torch.tensor([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
}
BONUS_THRESH = 0.3

def reward_function(paths):
    pass

def termination_function(paths):
    pass


