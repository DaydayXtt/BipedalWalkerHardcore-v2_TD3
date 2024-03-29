#!/usr/bin/env python
# coding=utf-8

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import datetime
import torch
import argparse
import gym

from td3 import TD3
from common.utils import plot_rewards

import numpy as np


def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='TD3', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='BipedalWalkerHardcore-v3', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=4000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=100, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=30000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=5e-3, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)

    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--expl_noise', default=0.25, type=float)

    parser.add_argument('--model_path_use',  # use the model used
                        default=curr_path + "/outputs/" + parser.parse_args().env_name +
                                '/' + "20240306-145944" + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    return args


def env_agent_config(cfg_, human=0):

    if human:
        env_ = gym.make(cfg_.env_name, render_mode="human")
    else:
        env_ = gym.make(cfg_.env_name)


    n_states = env_.observation_space.shape[0]
    n_actions = env_.action_space.shape[0]
    max_action = float(env_.action_space.high[0])

    env_with_Dead = False
    agent_ = TD3(n_states, n_actions, max_action, cfg_, env_with_Dead)
    return env_, agent_

def test(cfg_, env_, agent_):
    print('Start testing')
    print(f'Env:{cfg_.env_name}, Algorithm:{cfg_.algo_name}, Device:{cfg_.device}')

    rewards_ = []
    ma_rewards_ = []
    for i_ep in range(cfg_.test_eps):
        state = env_.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1

            env_.render()  # draw

            action = agent_.choose_action(state)
            next_state, reward, done, _ = env_.step(action)

            ep_reward += reward
            state = next_state

        print(f"Episode:{i_ep + 1}/{cfg_.test_eps}, Reward:{ep_reward:.1f}, Step:{i_step}")

        rewards_.append(ep_reward)
        if ma_rewards_:
            ma_rewards_.append(0.9 * ma_rewards_[-1] + 0.1 * ep_reward)
        else:
            ma_rewards_.append(ep_reward)
    print('Finish testing!')
    return rewards_, ma_rewards_


if __name__ == "__main__":
    cfg = get_args()

    # np.random.seed(1)
    # torch.manual_seed(1)

    # testing
    env, agent = env_agent_config(cfg, human=1)
    agent.load(path=cfg.model_path_use)
    rewards, ma_rewards = test(cfg, env, agent)
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
