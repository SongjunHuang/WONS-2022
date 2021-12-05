import math

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional


class Robot(object):
    def __init__(self, pos, angle):
        self.pos = pos
        self.angle = angle
        self.trajectory = []


class GridEnv(object):
    def __init__(self, world, discrete, n_agents, max_step, step_size):
        self.world = world
        self.discrete = discrete
        self.n_agents = n_agents
        self.max_step = max_step
        self.step_size = step_size


        # self.observation_space = 2 * self.n_agents
        self.observation_space = [7 * 7 + 2 for _ in range(self.n_agents)]
        # offset and angle for each agent
        self.action_space = [12 for _ in range(self.n_agents)]

        self.agents = None
        self.last_occupancy_map = None
        self.last_detections = None
        self.steps = 1
        self.reset()

    def random_pos(self, n):
        if self.discrete:
            x = np.random.randint(self.world.xw, size=n)
            y = np.random.randint(self.world.yw, size=n)
        else:
            x = np.random.rand(n) * self.world.xw
            y = np.random.rand(n) * self.world.yw
        return np.stack([x, y], axis=1)

    def ego_centric_grid(self, pos):
        # get 7 * 7 ego-centric map
        map = np.ones([self.world.occupancy_map.shape[0] + 2 * 4, self.world.occupancy_map.shape[1] + 2 * 4])
        map[4:-4, 4:-4] = self.world.occupancy_map
        ix = int(np.floor((pos[0] + 3 - self.world.minx) / self.world.xyreso))
        iy = int(np.floor((pos[1] + 3 - self.world.miny) / self.world.xyreso))
        ego_centric = map[ix - 3: ix + 4, iy - 3: iy + 4]
        return ego_centric.reshape(-1)

    def obs(self):
        # observation = [agent.pos.reshape((1, -1)) for agent in self.agents]
        # observation = [self.ego_centric_grid(agent.pos) for agent in self.agents]
        # all_pos = np.concatenate([self.agents[i].pos / 10 for i in range(self.n_agents)])
        # all_grids = [self.ego_centric_grid(self.agents[i].pos) for i in range(self.n_agents)]
        # observations = [np.concatenate(all_pos + all_grids) for _ in range(self.n_agents)]
        observations = []
        for i in range(self.n_agents):
            observation = np.concatenate([self.agents[i].pos / 10, self.ego_centric_grid(self.agents[i].pos)])
            observations.append(observation)
        return observations

    def reset(self):
        self.world.reset()
        if self.discrete:
            self.agents = [Robot((i, i), 0) for i in range(self.n_agents)]
        else:
            # self.agents = [Robot(np.random.rand(2), 0) for _ in range(self.n_agents)]
            self.agents = [Robot(np.asarray([i * 0.5, i * 0.5]), 0) for i in range(self.n_agents)]
        self.last_occupancy_map = self.world.occupancy_map.copy()
        self.last_detections = np.zeros(self.n_agents)
        self.steps = 1
        return self.obs()

    def reward(self, exploration_counts, collisions, out_of_bounds):
        # detected = np.sum(self.world.detected())
        # new_explored = np.sum(self.world.occupancy_map - self.last_occupancy_map)
        # return [new_explored / self.last_occupancy_map.shape[0] * self.last_occupancy_map.shape[1]
        #         for _ in range(self.n_agents)]
        # return exploration_count / self.world.xw * self.world.yw
        # return np.sum(detected) / self.steps
        exploration_rewards = exploration_counts / (self.steps + 1)
        rewards = exploration_rewards - 100 * collisions - 100 * out_of_bounds
        # self.last_detections = detected
        return rewards

    def collision(self):
        collisions = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.sqrt(np.sum((self.agents[i].pos - self.agents[j].pos) ** 2)) < 0.1:
                    collisions[i] = 1
                    collisions[j] = 1
        return collisions

    def out_of_bound(self):
        outs = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if self.agents[i].pos[0] >= self.world.maxx or self.agents[i].pos[0] <= self.world.minx \
                    or self.agents[i].pos[1] >= self.world.maxy or self.agents[i].pos[1] <= self.world.miny:
                outs[i] = 1
        return outs

    def done(self, collisions, out_of_bounds):
        if np.sum(self.world.detected()) == self.world.n_targets:
            return [True for _ in range(self.n_agents)]
        elif len(self.agents[0].trajectory) > self.max_step:
            return [len(self.agents[i].trajectory) > self.max_step for i in range(self.n_agents)]
        else:
            temp = collisions
            return [False if entry < 1 else True for entry in temp]

    def step(self, actions):
        exploration_counts = np.zeros(self.n_agents)
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            agent.trajectory.append(agent.pos)
            maxx, maxy, minx, miny = self.world.maxx, self.world.maxy, self.world.minx, self.world.miny
            # action = np.argmax(action)
            offset, angle_offset = None, None
            if action == 0:
                offset = np.asarray([-self.step_size, 0])  # left
                angle_offset = 0
            if action == 1:
                offset = np.asarray([self.step_size, 0])  # right
                angle_offset = 0
            if action == 2:
                offset = np.asarray([0, self.step_size])  # forward
                angle_offset = 0
            if action == 3:
                offset = np.asarray([0, -self.step_size])  # backward
                angle_offset = 0
            if action == 4:
                offset = np.asarray([-self.step_size, 0])  # left
                angle_offset = 120
            if action == 5:
                offset = np.asarray([self.step_size, 0])  # right
                angle_offset = 120
            if action == 6:
                offset = np.asarray([0, self.step_size])  # forward
                angle_offset = 120
            if action == 7:
                offset = np.asarray([0, -self.step_size])  # backward
                angle_offset = 120
            if action == 8:
                offset = np.asarray([-self.step_size, 0])  # left
                angle_offset = 240
            if action == 9:
                offset = np.asarray([self.step_size, 0])  # right
                angle_offset = 240
            if action == 10:
                offset = np.asarray([0, self.step_size])  # forward
                angle_offset = 240
            if action == 11:
                offset = np.asarray([0, -self.step_size])  # backward
                angle_offset = 240
            agent.pos = np.clip(agent.pos + offset, [minx + 1, miny + 1], [maxx - 1, maxy - 1])
            agent.angle = angle_offset
            # print(agent.pos, agent.angle)
            exploration_count = self.world.generate_ray_casting_grid_map(agent.pos[0], agent.pos[1], agent.angle)
            exploration_counts[i] = exploration_count
        collisions = self.collision()
        out_of_bounds = self.out_of_bound()
        rewards = self.reward(exploration_counts, collisions, out_of_bounds)
        dones = self.done(collisions, out_of_bounds)
        self.last_occupancy_map = self.world.occupancy_map.copy()
        self.steps += 1
        # print("{}/{} targets detected".format(sum(self.world.detected()), self.world.n_targets))
        return self.obs(), rewards, dones

    def visualize(self):
        # x, y = np.mgrid[slice(-self.world.grid_width / 2, self.world.grid_width / 2, self.world.xyreso),
        #                 slice(-self.world.grid_height / 2, self.world.grid_height / 2, self.world.xyreso)]
        y, x = np.meshgrid(np.arange(-self.world.grid_width / 2,
                                     self.world.grid_width / 2 + self.world.xyreso,
                                     self.world.xyreso),
                           np.arange(-self.world.grid_height / 2,
                                     self.world.grid_height / 2 + self.world.xyreso,
                                     self.world.xyreso))
        agent_x = [agent.pos[0] for agent in self.agents]
        agent_y = [agent.pos[1] for agent in self.agents]
        plt.plot(agent_x, agent_y, "ob")
        plt.plot(self.world.target_x, self.world.target_y, 'rx')
        plt.plot(self.world.obstacle_x, self.world.obstacle_y, 'gs')

        plt.pcolor(x, y, self.world.occupancy_map, vmax=1.0, cmap=plt.cm.Blues)
        plt.xlim((-self.world.grid_width / 2, self.world.grid_width / 2))
        plt.ylim((-self.world.grid_height / 2, self.world.grid_height / 2))
        plt.show()
