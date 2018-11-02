import torch
import torch.nn as nn
import numpy as np

from envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def calc_modes(rewards, next_values, pre_values, evaluations, mode, tonic_g, phasic_g, masks):
    rewards = rewards.cpu().squeeze().numpy()
    next_values = next_values.cpu().squeeze().numpy()
    new_pre_values = []
    new_evaluations = []
    for i in range(len(pre_values)):
        if masks[i] == 0:
            new_evaluations.append(0)
            new_pre_values.append(None)
        elif pre_values[i] == None:
            new_evaluations.append(0)
            new_pre_values.append(next_values[i])
        else:
            new_evaluations.append(0.75*evaluations[i]+0.25*(next_values[i]+rewards[i]-pre_values[i]))
            new_pre_values.append(next_values[i])

    # mode 1: use diff; mode 0: use abs(diff)
    if mode == 1:
        g = torch.FloatTensor([[tonic_g] if evaluation < 2.0 else [phasic_g] for evaluation in evaluations])
    else:
        g = torch.FloatTensor([[tonic_g] if abs(evaluation) > 2.0  else [phasic_g] for evaluation in evaluations])
    
    return new_evaluations, g, new_pre_values

def tanh_g(x,g):
    x = x/g
    return torch.tanh(x)

def neural_activity(s, g, mid = 128):
    assert(s.shape[0] == g.shape[0])
    for i in range(s.shape[0]):
        s[i] = (torch.tanh((s[i]-mid)/g[i])+1)/2
    return s

