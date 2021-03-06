import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from config import Config

from math import exp

from utils.utils import to_tensor


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                 in_features,
                 num_actions,
                 num_options,
                 temperature=1.0,
                 eps_start=1.0,
                 eps_min=0.1,
                 eps_decay=int(100),
                 eps_test=0.05,
                 device='cuda',
                 testing=False):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination), next_option

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.detach(), logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        # print(Q.shape)
        return Q.argmax(dim=-1)

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model, model_prime, data_batch):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(to_tensor(obs)).squeeze(0)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(to_tensor(next_obs)).squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states = model.get_state(to_tensor(next_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * 0.99 * \
         ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob *
          next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actor_loss(obs, options, logps, entropys, rewards, dones, next_obss, models, model_primes):
    state = models.get_state(to_tensor(obs))
    next_state = models.get_state(to_tensor(next_obss))
    next_state_prime = model_primes.get_state(to_tensor(next_obss))

    option_term_prob = models.get_terminations(state)[:, options]
    next_option_term_prob = models.get_terminations(next_state)[:, options].detach()

    Q = models.get_Q(state).detach().squeeze()
    next_Q_prime = model_primes.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = rewards + (1 - dones) * 0.99 * \
         ((1 - next_option_term_prob) * next_Q_prime[options] + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (
            Q[options].detach() - Q.max(dim=-1)[0].detach() + 0.01) * (1 - dones)

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logps * (gt.detach() - Q[options]) - 0.01 * entropys

    actor_loss_value = termination_loss + policy_loss
    return actor_loss_value


def merge_critics(models):
    n_model = len(models)
    state_dict = models[0].state_dict().copy()
    for key in state_dict:
        state_dict[key] = state_dict[key] / n_model
    for key in state_dict:
        for model in models[1:]:
            state_dict[key] = state_dict[key] + model.state_dict()[key] / n_model
    for model in models:
        model.load_state_dict(state_dict)