from comet_ml import Experiment

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

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot

import collections

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'



if args.comet == "offline":
    experiment = OfflineExperiment(project_name="recurrent-filter", workspace="nishanthvanand",
    disabled=args.disable_log, offline_directory="/scratch/nish127/comet_offline",
    parse_args=False)
elif args.comet == "online":
    experiment = experiment = Experiment(api_key="tSACzCGFcetSBTapGBKETFARf",
                        project_name="recurrent-filter", workspace="nishanthvanand",disabled=args.disable_log,
                        parse_args=False)

experiment.log_parameters(vars(args))

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False, args.num_frame_stack)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'est_filter':args.est_filter, 'filter_mem':args.filter_memory})
    actor_critic.to(device)

    '''
    passing the size of latent representation here
    '''
    if len(envs.observation_space.shape)==3:
        hidden_size = 512
    elif len(envs.observation_space.shape)==1:
        hidden_size = 64
    else:
        NotImplementedError

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               lr_filter=args.lr_filter, reg_filter=args.reg_filter, filter_mem=args.filter_memory,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape,
                        hidden_size, (1, args.filter_memory),
                        envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()

    filter_coeff_list = []

    for j in range(num_updates):
    
        value_prev = collections.deque([torch.zeros(args.num_processes, 1).to(device) for i in range(args.filter_memory)], maxlen=args.filter_memory)
        value_prev_eval = collections.deque([torch.zeros(args.num_processes, 1).to(device) for i in range(args.filter_memory)], maxlen=args.filter_memory)

        #filter_mem_latent = collections.deque([torch.zeros(args.num_processes, hidden_size).to(device) for i in range(args.filter_memory)], maxlen=args.filter_memory)
        filter_mem_latent_eval = collections.deque([torch.zeros(args.num_processes, hidden_size).to(device) for i in range(args.filter_memory)], maxlen=args.filter_memory)

        if args.filter_type == "IIR":
            raise NotImplementedError

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        value_prev=value_prev,
                        filter_type=args.filter_type)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value, next_latent = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1])

        next_value = next_value.detach()
        next_latent = next_latent.detach()

        rollouts.compute_returns(next_value, next_latent, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, value_prev_eval, filter_mem_latent_eval, att_list = agent.update(rollouts, value_prev_eval, filter_mem_latent_eval, filter_type=args.filter_type)

        filter_coeff_list.append(att_list)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            if args.est_filter:
                torch.save(save_model, os.path.join(save_path, args.env_name +"_seed_"+str(args.seed) + "_beta_est.pt"))
            else:
                torch.save(save_model, os.path.join(save_path, args.env_name +"_seed_"+str(args.seed) + "_no_beta.pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

            filter_coeff_mean = {"coeff_mean_"+str(i):0 for i in range(args.filter_memory)}
            filter_coeff_std = {"coeff_std_"+str(i):0 for i in range(args.filter_memory)}

            filter_list = []
            for batches in range(len(filter_coeff_list)):
                    if not args.cuda:
                        filter_np = filter_coeff_list[batches].numpy()
                    else:
                        filter_np = filter_coeff_list[batches].cpu().numpy()
                    filter_list.append(filter_np.mean(1).ravel())
            filter_numpy = np.array(filter_list)
            filter_mean = filter_numpy.mean(0)
            filter_std = filter_numpy.std(0)

            for idx, (m,s) in enumerate(zip(filter_mean, filter_std)):
                filter_coeff_mean["coeff_mean_"+str(idx)] = m
                filter_coeff_std["coeff_std_"+str(idx)] = s

            experiment.log_metrics({"mean reward": np.mean(episode_rewards),
                                 "Value loss": value_loss, "Action Loss": action_loss},
                                 step=j * args.num_steps * args.num_processes)

            experiment.log_metrics(filter_coeff_mean, step=j * args.num_steps * args.num_processes)
            experiment.log_metrics(filter_coeff_std, step=j * args.num_steps * args.num_processes)

            del filter_coeff_list[:]

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                           for done_ in done],
                                           dtype=torch.float32,
                                           device=device)
                
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_env_steps)
            except IOError:
                pass


if __name__ == "__main__":
    main()
