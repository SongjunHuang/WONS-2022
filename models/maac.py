import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.gumble_softmax import *


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Tanh()
        )

    def forward(self, state):
        output = self.model(state)
        return output


# class Critic(nn.Module):
#     def __init__(self, state_space, action_space):
#         super(Critic, self).__init__()
#         self.state = nn.Linear(state_space, 256)
#         self.model = nn.Sequential(
#             nn.Linear(256 + action_space, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, state, action):
#         output = self.state(state)
#         output = self.model(torch.cat([output, action], dim=1))
#         return output


class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space + action_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        # output = self.state(state)
        output = self.model(torch.cat([state, action], dim=1))
        return output


class MADDPG:
    def __init__(self, state_size, action_size, n_agent, gamma=0.99, lr_actor=0.01, lr_critic=0.05,
                 EPS_START=1, EPS_END=0.1, EPS_DECAY=int(1e6), update_freq=200):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agent = n_agent
        self.gamma = gamma
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.update_freq = update_freq
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device {}".format(self.device))

        self.actors = [Actor(state_size, action_size).to(self.device) for _ in range(n_agent)]
        self.actors_target = [Actor(state_size, action_size).to(self.device) for _ in range(n_agent)]

        self.critics = [Critic(state_size * n_agent, 1 * n_agent).to(self.device) for _ in range(n_agent)]
        self.critics_target = [Critic(state_size * n_agent, 1 * n_agent).to(self.device) for _ in range(n_agent)]
        # self.critic = Critic(state_size * n_agent, action_size * n_agent).to(self.device)
        # self.critic_target = Critic(state_size * n_agent, action_size * n_agent).to(self.device)

        [actor_target.eval() for actor_target in self.actors_target]
        [critic_target.eval() for critic_target in self.critics_target]

        self.actors_optim = [optim.Adam(actor.parameters(), lr_actor) for actor in self.actors]
        self.critics_optim = [optim.Adam(critic.parameters(), lr_critic) for critic in self.critics]
        # self.actor_optim = optim.Adam(sum([list(actor.parameters()) for actor in self.actors]))
        # self.critic_optim = optim.Adam(sum([list(critic.parameters()) for critic in self.critics]))

        self.steps = 0

    def update_target(self):
        for i in range(self.n_agent):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs
        return torch.FloatTensor(inputs).to(self.device)

    def choose_action(self, states):
        # actions = [actor(self.to_tensor(state)).detach().cpu().numpy() for actor, state in zip(self.actors, states)]
        actions = [onehot_from_logits(actor(self.to_tensor(state).view(1, -1)), self.epsilon()).detach().cpu().numpy()
                   for actor, state in zip(self.actors, states)]
        actions = [np.argmax(action) for action in actions]
        return actions

    # def epsilon(self):
    #     return self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * self.steps / self.EPS_DECAY)

    def epsilon(self):
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-self.steps / self.EPS_DECAY)
        self.steps += 1
        return eps

    def learn(self, s, a, r, sn, d):
        states = [self.to_tensor(state) for state in s]
        actions = [self.to_tensor(action) for action in a]
        rewards = [self.to_tensor(reward) for reward in r]
        # print(rewards)
        rewards = [(reward - torch.mean(reward)) / torch.std(reward) for reward in rewards]
        # print(rewards)
        rewards = [torch.nan_to_num(reward, nan=0.0) for reward in rewards]
        # print([np.sum(rr) for rr in r], [torch.sum(rr).item() for rr in rewards])
        states_next = [self.to_tensor(state_next) for state_next in sn]
        dones = [self.to_tensor(done.astype(int)) for done in d]
        all_state = torch.cat(states, dim=1)
        all_action = torch.stack(actions, dim=1)
        all_state_next = torch.cat(states_next, dim=1)
        actor_losses = 0
        for i in range(self.n_agent):
            cur_action = all_action.clone()
            action = gumbel_softmax(self.actors[i](states[i]))
            action = torch.argmax(action, dim=1)
            # action_size = action.shape[1]
            cur_action[:, i] = action
            actor_loss = -torch.mean(self.critics[i](all_state, cur_action))
            actor_losses += actor_loss

        actions_next = [onehot_from_logits(actor_target(state_next), self.epsilon()).detach()
                        for state_next, actor_target in zip(states_next, self.actors_target)]
        actions_next = [torch.argmax(action_next, dim=1) for action_next in actions_next]
        all_action_next = torch.stack(actions_next, dim=1)
        critic_losses = 0
        for i in range(self.n_agent):
            next_value = self.critics_target[i](all_state_next, all_action_next)
            Q = self.critics[i](all_state, all_action)
            target = rewards[i] + self.gamma * next_value.detach()
            critic_loss = F.mse_loss(Q, target, reduction='mean')
            critic_losses += critic_loss
            # print(Q)

        # print(actor_losses, critic_losses)

        # actor
        # self.actor_optim.zero_grad()
        [actor_optim.zero_grad() for actor_optim in self.actors_optim]
        actor_losses.backward()
        [nn.utils.clip_grad_norm_(actor.parameters(), 0.5) for actor in self.actors]
        # self.actor_optim.step()
        [actor_optim.step() for actor_optim in self.actors_optim]
        # critic
        # self.critic_optim.zero_grad()
        [critic_optim.zero_grad() for critic_optim in self.critics_optim]
        critic_losses.backward()
        [nn.utils.clip_grad_norm_(critic.parameters(), 0.5) for critic in self.critics]
        # self.critic_optim.step()
        [critic_optim.step() for critic_optim in self.critics_optim]
        # update target networks
        if self.steps % self.update_freq == 0:
            self.update_target()
        self.steps += 1

        return (actor_losses + critic_losses).item()

    # def save_model(self, directory):
    #     save_content = {}
    #     for i in range(self.n_agent):
    #         save_content['actor_{}'.format(i)] = self.actors[i].state_dict()
    #         save_content['critic_{}'.format(i)] = self.critics[i].state_dict()
    #         save_content['actor_optimizer_{}'.format(i)] = self.actors_optim[i].state_dict()
    #         save_content['critic_optimizer_{}'.format(i)] = self.critics_optim[i].state_dict()
    #     torch.save(save_content, directory + "MAAC_model")
    #
    # def load_model(self, directory):
    #     saved_content = torch.load(directory + "MAAC_model")
    #     for i in range(self.n_agent):
    #         self.actors[i].load_state_dict(saved_content['actor_{}'.format(i)])
    #         self.critics[i].load_state_dict(saved_content['critic_{}'.format(i)])
    #         self.actors_optim[i].load_state_dict(saved_content['actor_optimizer_{}'.format(i)])
    #         self.critics_optim[i].load_state_dict(saved_content['critic_optimizer_{}'.format(i)])
    #     self.update_target()