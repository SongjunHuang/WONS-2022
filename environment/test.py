from environment.grid import GridEnv
from environment.world import GridWorld
import numpy as np

np.random.seed(2021)

grid_width = 20
grid_height = 20
fov = 60
xyreso = 1
yawreso = 5
sensing_range = 8
n_obstacles = 10
n_targets = 5
n_agents = 3

world = GridWorld(grid_width=grid_width, grid_height=grid_height, fov=fov, xyreso=xyreso, yawreso=yawreso,
                  sensing_range=sensing_range, n_targets=n_targets)
env = GridEnv(world, discrete=False, n_agents=n_agents, max_step=100)
for _ in range(5):
    actions = []
    for i in range(n_agents):
        random_action = np.random.rand(3)
        random_action[:2] = (random_action[:2] - 0.5) * 4
        random_action[2] = (random_action[2] - 0.5) * 360
        actions.append(random_action)
    env.step(actions)
    env.visualize()