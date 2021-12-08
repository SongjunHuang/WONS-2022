import numpy as np
import time
import random
import os
import torch

from copy import deepcopy
from environment.grid import GridEnv
from environment.world import GridWorld
from models.option_critic import OptionCriticFeatures
from models.option_critic import actor_loss as actor_loss_fn
from models.option_critic import critic_loss as critic_loss_fn
from models.option_critic import merge_critics
# from models.maac_seperate import MADDPG
# from utils.memory import Memory
from utils.experience_replay import ReplayBuffer
from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from utils import general_utilities as general_utilities
from utils.utils import to_tensor
from config import Config


def play(is_testing):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["exploration_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["collision_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_theta_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_mu_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_sigma_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_dt_{}".format(i) for i in range(env.n_agents)])
    statistics_header.extend(["ou_x0_{}".format(i) for i in range(env.n_agents)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(statistics_header)

    option_critic_prime = [deepcopy(option_critic[i]) for i in range(Config.n_agents)]
    optim = [torch.optim.RMSprop(option_critic[i].parameters(), lr=Config.lr_actor) for i in range(Config.n_agents)]

    sum_rewards = 0
    for episode in range(Config.episodes):
        option_lengths = [{opt: [] for opt in range(Config.num_options)} for _ in range(Config.n_agents)]
        states, greedy_options = [], []
        obs = env.reset()
        for i in range(Config.n_agents):
            state = option_critic[i].get_state(to_tensor(obs[i]))
            greedy_option = option_critic[i].greedy_option(state)
            states.append(state)
            greedy_options.append(greedy_option)

        current_options = np.zeros(Config.n_agents, dtype=np.int)
        episode_losses = np.zeros(env.n_agents)
        episode_rewards = np.zeros(env.n_agents)
        collision_count = np.zeros(env.n_agents)
        steps = 0
        option_termination = [True for _ in range(Config.n_agents)]
        curr_op_len = np.zeros(Config.n_agents, dtype=np.int)
        while True:
            steps += 1
            actions, logps, entropys, epsilons = [], [], [], []
            # next_obs, rewards, dones = [], [], []
            for i in range(Config.n_agents):
                epsilon = option_critic[i].epsilon
                epsilons.append(epsilon)
                if option_termination[i]:
                    option_lengths[i][current_options[i]].append(curr_op_len)
                    current_options[i] = np.random.choice(Config.num_options) if np.random.rand() < epsilons[i] else \
                        greedy_options[i]
                    curr_op_len[i] = 0
                # act
                action, logp, entropy = option_critic[i].get_action(states[i], current_options[i])
                actions.append(action)
                logps.append(logp)
                entropys.append(entropy)

            # step
            next_ob, reward, done = env.step(actions)
            for i in range(Config.n_agents):
                memories[i].push(obs[i], current_options[i], reward[i], next_ob[i], done[i])
                states[i] = option_critic[i].get_state(to_tensor(next_ob[i])).detach()
                option_termination[i], greedy_options[i] = option_critic[i].predict_option_termination(states[i],
                                                                                                       current_options[i])
                if not is_testing:
                    if len(memories[0]) > Config.batch_size * 10:
                        actor_loss = actor_loss_fn(obs[i], current_options[i], logps[i], entropys[i],
                                                   reward[i], done[i], next_ob[i], option_critic[i],
                                                   option_critic_prime[i])
                        # L = 0
                        # models = option_critic[i].options_W
                        # L += torch.abs(models[current_options[i], :, :] - models[0, :, :]) \
                        #      + torch.abs(models[current_options[i], :, :] - models[1, :, :]) \
                        #      + torch.abs(models[current_options[i], :, :] - models[2, :, :])
                        #
                        # actor_losss += L

                        loss = actor_loss
                        if steps % 4 == 0:
                            data_batch = memories[i].sample(Config.batch_size)
                            critic_loss = critic_loss_fn(option_critic[i], option_critic_prime[i], data_batch)
                            loss += critic_loss

                        optim[i].zero_grad()
                        loss.backward()
                        optim[i].step()
                        episode_losses[i] += loss.item()
                    else:
                        episode_losses[i] = -1
            # if steps % 10 == 0:
            #     merge_critics([option_critic[i].features for i in range(env.n_agents)])
            #     merge_critics([option_critic[i].Q for i in range(env.n_agents)])
            obs = next_ob
            episode_rewards += reward

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n_agents)])
                statistic.extend([episode_losses[i] for i in range(env.n_agents)])
                statistic.extend([np.sum(env.world.occupancy_map > 0) for _ in range(env.n_agents)])
                statistic.extend([0 for _ in range(env.n_agents)])
                statistic.extend([actors_noise[i].theta for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].mu for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].sigma for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].dt for i in range(env.n_agents)])
                statistic.extend([actors_noise[i].x0 for i in range(env.n_agents)])
                statistics.add_statistics(statistic)
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                    # env.visualize()
                break

        if episode % Config.checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(Config.csv_filename_prefix, episode))
    print("Avg rewards: {}".format(sum_rewards / Config.episodes / Config.n_agents))
    return statistics


if __name__ == '__main__':
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)
    torch.cuda.manual_seed(Config.random_seed)

    for n_agent in [4, 6, 8]:
        Config.n_agents = n_agent
        Config.update()
        print(Config.n_agents, Config.scheme, Config.comm_fail_prob)
        print('Start experiment for scheme {}'.format(Config.scheme))
        # print("running experiment for {} agents".format(n_agent))
        if not os.path.exists(Config.experiment_prefix + Config.scheme):
            os.makedirs(Config.experiment_prefix + Config.scheme)
        for rounds in range(5):
            # init env
            world = GridWorld(Config.grid_width, Config.grid_height, Config.fov, Config.xyreso, Config.yawreso,
                              Config.sensing_range, Config.n_targets)
            env = GridEnv(world, discrete=Config.discrete, n_agents=Config.n_agents, max_step=Config.max_step,
                          step_size=Config.step_size)

            # Extract ou initialization values
            ou_mus = [np.zeros(env.action_space[i]) for i in range(env.n_agents)]
            ou_sigma = [0.3 for i in range(env.n_agents)]
            ou_theta = [0.15 for i in range(env.n_agents)]
            ou_dt = [1e-2 for i in range(env.n_agents)]
            ou_x0 = [None for i in range(env.n_agents)]

            # set random seed

            option_critic = [OptionCriticFeatures(
                in_features=env.observation_space[0],
                num_actions=env.action_space[0],
                num_options=Config.num_options
            ) for _ in range(Config.n_agents)]
            actors_noise = []
            memories = []
            for i in range(env.n_agents):
                n_action = env.action_space[i]
                state_size = env.observation_space[i]
                speed = 1

                actors_noise.append(OrnsteinUhlenbeckActionNoise(
                    mu=ou_mus[i],
                    sigma=ou_sigma[i],
                    theta=ou_theta[i],
                    dt=ou_dt[i],
                    x0=ou_x0[i]))
                buffer = ReplayBuffer(Config.memory_size)
                memories.append(buffer)

            start_time = time.time()
            # play
            statistics = play(is_testing=False)
            # maddpgs.save_model("../results/model/")
            # bookkeeping
            print("Finished {} episodes in {} seconds".format(Config.episodes, time.time() - start_time))
            # tf.summary.FileWriter(args.experiment_prefix +
            #                       args.weights_filename_prefix, session.graph)
            # save_path = saver.save(session, os.path.join(
            #     args.experiment_prefix + args.weights_filename_prefix, "models"), global_step=args.episodes)
            save_path = Config.experiment_prefix + Config.scheme + '/' + Config.csv_filename_prefix + "_{}.csv".format(
                rounds)
            statistics.dump(save_path)
            print("saving model to {}".format(save_path))
