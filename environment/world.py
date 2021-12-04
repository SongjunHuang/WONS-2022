import math
import numpy as np
import matplotlib.pyplot as plt


class precastDB:
    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.ix = 0
        self.iy = 0

    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle)


class GridWorld(object):
    def __init__(self, grid_width, grid_height, fov, xyreso, yawreso, sensing_range, n_targets):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fov = fov
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.sensing_range = sensing_range
        self.n_targets = n_targets

        self.calc_grid_map_config(grid_width, grid_height)
        # assert (self.grid_width == self.xw) and (self.grid_height == self.yw)
        self.occupancy_map = None

        self.obstacle_x, self.obstacle_y = None, None
        self.target_x, self.target_y = None, None
        self.reset()

    def get_rect_coordinates(self, bottom, left, width):
        coordinates = []
        for i in range(bottom, bottom + width + 1):
            for j in range(left, left + width + 1):
                coordinates.append([i, j])
        return np.asarray(coordinates)

    def reset(self):
        # self.obstacle_x = (np.random.rand(self.n_obstacles) - 0.5) * self.grid_width
        # self.obstacle_y = (np.random.rand(self.n_obstacles) - 0.5) * self.grid_height
        obstacle1 = self.get_rect_coordinates(-7, 2, 5)
        obstacle2 = self.get_rect_coordinates(-7, -7, 5)
        obstacle3 = self.get_rect_coordinates(2, -7, 5)
        obstacles = np.concatenate([obstacle1, obstacle2, obstacle3], axis=0)
        self.obstacle_x = obstacles[:, 0]
        self.obstacle_y = obstacles[:, 1]
        self.n_obstacles = len(obstacles)

        # self.target_x = (np.random.rand(self.n_targets) - 0.5) * (self.grid_width - 5)
        # self.target_y = (np.random.rand(self.n_targets) - 0.5) * (self.grid_height - 5)
        targets = np.asarray([[-8, 5],
                              [-6, 8],
                              [-8, -5],
                              [-5, -8],
                              [0, -4],
                              [4, -9],
                              [8, -4],
                              [3, 7],
                              [6, 3],
                              [7, 7]])
        self.target_x = targets[:, 0]
        self.target_y = targets[:, 1]
        self.occupancy_map = np.zeros([self.xw + 1, self.yw + 1], dtype=int)
        self.obstacle_map = np.zeros([self.xw, self.yw])
        obstacle_ix = (self.obstacle_x - self.minx) / self.xyreso
        obstacle_iy = (self.obstacle_y - self.miny) / self.xyreso
        self.obstacle_map[obstacle_ix.astype(int), obstacle_iy.astype(int)] = 1

    def calc_grid_map_config(self, grid_width, grid_height):
        # minx = round(min(ox) - EXTEND_AREA / 2.0)
        # miny = round(min(oy) - EXTEND_AREA / 2.0)
        # maxx = round(max(ox) + EXTEND_AREA / 2.0)
        # maxy = round(max(oy) + EXTEND_AREA / 2.0)
        self.minx = np.floor(0 - grid_width / 2.0)
        self.miny = np.floor(0 - grid_height / 2.0)
        self.maxx = np.floor(grid_width / 2.0) - 1
        self.maxy = np.floor(grid_height / 2.0) - 1
        self.xw = int(np.floor(grid_width / self.xyreso))
        self.yw = int(np.floor(grid_height / self.xyreso))


    def atan_zero_to_twopi(self, y, x):
        angle = math.atan2(y, x)
        if angle < 0.0:
            angle += math.pi * 2.0
        return np.rad2deg(angle)

    def precasting(self, agent_x, agent_y):
        precast = [[] for _ in range(int(np.floor(360 / self.yawreso)) + 1)]
        obstacle_onpath = [[] for _ in range(int(np.floor(360 / self.yawreso)) + 1)]

        for ix in range(self.xw):
            for iy in range(self.yw):
                px = ix * self.xyreso + self.minx
                py = iy * self.xyreso + self.miny

                d = math.hypot(px - agent_x, py - agent_y)
                angle = self.atan_zero_to_twopi(py - agent_y, px - agent_x)

                angleid = int(math.floor(angle / self.yawreso))

                pc = precastDB()

                pc.px = px
                pc.py = py
                pc.d = d
                pc.ix = ix
                pc.iy = iy
                pc.angle = angle

                precast[angleid].append(pc)
                if self.obstacle_map[ix, iy] == 1:
                    obstacle_onpath[angleid].append([px, py])

        return precast, obstacle_onpath

    def generate_ray_casting_grid_map(self, agent_x, agent_y, agent_angle_deg):
        exploration_count = 0
        precast, obstacle_onpath = self.precasting(agent_x, agent_y)

        start_angle, end_angle = agent_angle_deg, agent_angle_deg + self.fov
        start_angleid = int(math.floor(start_angle % 360 / self.yawreso))
        end_angleid = int(math.floor(end_angle % 360 / self.yawreso))
        end_angleid = end_angleid + 1 if end_angle > start_angleid else end_angleid - 1
        step = 1 if start_angle < end_angle else -1
        for angleid in range(start_angleid, end_angleid, step):
            for grid in precast[angleid]:
                agent2grid_dist = math.hypot(grid.px - agent_x, grid.py - agent_y)
                if agent2grid_dist <= self.sensing_range:
                    if len(obstacle_onpath[angleid]) > 0:
                        agent2obstacle_dists = [math.hypot(obstacle_onpath[angleid][i][0] - agent_x,
                                                           obstacle_onpath[angleid][i][1] - agent_y)
                                                for i in range(len(obstacle_onpath[angleid]))]
                        agent2obstacle_dist = np.min(agent2obstacle_dists)
                        if agent2grid_dist < agent2obstacle_dist:
                            if self.occupancy_map[grid.ix, grid.iy] < 1:
                                self.occupancy_map[grid.ix, grid.iy] = 1
                                exploration_count += 1
                            # else:
                            #     exploration_count -= 1
                    else:
                        if self.occupancy_map[grid.ix, grid.iy] < 1:
                            self.occupancy_map[grid.ix, grid.iy] = 1
                            exploration_count += 1
                        # else:
                        #     exploration_count -= 1
        return exploration_count

    def detected(self):
        targets_detected = np.zeros(self.n_targets)
        for i in range(self.n_targets):
            x, y = self.target_x[i], self.target_y[i]
            ix = int(np.floor((x - self.minx) / self.xyreso))
            iy = int(np.floor((y - self.miny) / self.xyreso))
            if self.occupancy_map[ix, iy] > 0:
                targets_detected[i] = 1
        return targets_detected

