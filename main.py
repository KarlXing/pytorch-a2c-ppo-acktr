import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize, calc_modes, neural_activity
from visualize import visdom_plot
from tensorboardX import SummaryWriter

args = get_args()

# assert args.algo in ['a2c', 'ppo', 'acktr']
assert args.algo == 'a2c'
# if args.recurrent_policy:
#     assert args.algo in ['a2c', 'ppo'], \
#         'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("config:\n")
    print("activation:", args.activation)
    print("evaluation:", args.evaluation)
    print("evaluation mode:", args.evaluation_mode)
    print("evaluation layer:", args.evaluation_layer)
    writer = SummaryWriter()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}, activation = args.activation, modulation = args.evaluation)
    # load trained model
    if args.load_model_path != None:
        state_dicts = torch.load(args.load_model_path)
        actor_critic.load_nets(state_dicts)

    actor_critic.to(device)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':
    #     agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
    #                      args.value_loss_coef, args.entropy_coef, lr=args.lr,
    #                            eps=args.eps,
    #                            max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
    #                            args.entropy_coef, acktr=True)


    tonic_g = 1
    phasic_g = 1
    if args.evaluation and args.evaluation_layer == 1:  # f1 modulation
        tonic_g = args.f1_tonic_g
        phasic_g = args.f1_phasic_g
    if args.evaluation and args.evaluation_layer == 0:  # input activation
        tonic_g = args.input_tonic_g
        phasic_g = args.input_phasic_g

    g = torch.ones(args.num_processes,1)*tonic_g
    g_device = (torch.ones(args.num_processes,1)*tonic_g).to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, tonic_g)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    pre_value = [None for i in range(args.num_processes)]
    evaluations = [0 for i in range(args.num_processes)]
    ## to calculate next_value and update g
    next_recurrent_hidden_states = torch.zeros(args.num_processes, actor_critic.recurrent_hidden_state_size).to(device)
    next_g = torch.zeros(args.num_processes,1).to(device)
    next_masks = torch.zeros(args.num_processes,1).to(device)
    next_obs = torch.zeros(args.num_processes, *envs.observation_space.shape).to(device)

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.g[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            # calculate next value with old g and decide new g
            if args.evaluation or args.evaluation_log:
                if args.evaluation_layer == 0:
                    next_obs.copy_(neural_activity(obs,g_device))
                else:
                    next_obs.copy_(obs/255)
                next_recurrent_hidden_states.copy_(recurrent_hidden_states)
                next_g.copy_(g)
                next_masks.copy_(masks)
                with torch.no_grad():
                    next_value = actor_critic.get_value(next_obs,
                                                next_g,
                                                next_recurrent_hidden_states,
                                                next_masks).detach()
                evaluations, g, pre_value = calc_modes(reward, next_value, pre_value, evaluations, args.evaluation_mode, tonic_g, phasic_g, masks)
                g_device.copy_(g)

            # observation processing with new g
            if args.evaluation and args.evaluation_layer == 0:
                obs = neural_activity(obs, g_device)
            else:
                obs = obs/255.0

            for idx in range(len(infos)):
                info = infos[idx]
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    steps_done = j*args.num_steps*args.num_processes + step*args.num_processes + idx
                    writer.add_scalar('data/reward', info['episode']['r'], steps_done )
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, g)

            # record evaluation value to help decide parameters to switch modes
            if args.evaluation_log:
                writer.add_scalar('data/evaluations', evaluations[0], j*args.num_steps*args.num_processes + step*args.num_processes)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.g[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                    pass

            state_dicts = actor_critic.save_nets()
            torch.save(state_dicts, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    main()
