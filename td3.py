#!/usr/bin/env python
# coding=utf-8
"""
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2022-06-09 19:04:44
@Discription: 
@Environment: python 3.7.7
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


class ReplayBuffer(object):
    def __init__(self, n_states, n_actions, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, n_states))
        self.action = np.zeros((max_size, n_actions))
        self.next_state = np.zeros((max_size, n_states))
        self.reward = np.zeros((max_size, 1))
        self.dead = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, next_state, reward, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.dead[self.ptr] = dead
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, max_action, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return self.max_action * x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(sa))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        return q1


class TD3:
    def __init__(self, n_states, n_actions, max_action, cfg, env_with_Dead):
        self.device = cfg.device

        self.critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.actor = Actor(n_states, n_actions, cfg.hidden_dim, max_action).to(cfg.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # soft update the target a/c
        self.gamma = cfg.gamma

        self.max_action = max_action
        self.total_it = 0
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = cfg.policy_freq
        self.memory = ReplayBuffer(n_states, n_actions)
        self.env_with_Dead = env_with_Dead

        self.expl_noise = cfg.expl_noise

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def update(self):
        self.total_it += 1

        # smapling a batch of records from buffer
        state, action, next_state, reward, dead_mask = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                    self.target_actor(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            '''DEAD OR NOT'''
            if self.env_with_Dead:
                target_Q = reward + (1 - dead_mask) * self.gamma * target_Q
            else:
                target_Q = reward + self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update the terget critic and actor
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) +
                    param.data * self.soft_tau
                )

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic", map_location=torch.device(self.device)))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer", map_location=torch.device(self.device)))
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor", map_location=torch.device(self.device)))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer", map_location=torch.device(self.device)))
        self.target_actor = copy.deepcopy(self.actor)
