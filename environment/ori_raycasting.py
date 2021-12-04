import math
import numpy as np
import matplotlib.pyplot as plt

EXTEND_AREA = 10.0

agent_x = 3
agent_y = -3
agent_angle_deg = math.radians(90)
agent_sight_angle_deg = math.radians(60)
agent_sensing_range = 3


show_animation = True


def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


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


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0

    return angle


def precasting(minx, miny, xw, yw, xyreso, yawreso):

    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]

    for ix in range(xw):
        for iy in range(yw):
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            d = math.hypot(px - agent_x, py - agent_y)
            angle = atan_zero_to_twopi(py - agent_y, px - agent_x)

            angleid = int(math.floor(angle / yawreso))

            pc = precastDB()

            pc.px = px
            pc.py = py
            pc.d = d
            pc.ix = ix
            pc.iy = iy
            pc.angle = angle

            precast[angleid].append(pc)

    return precast


def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso):

    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    precast = precasting(minx, miny, xw, yw, xyreso, yawreso)
    #
    # for (x, y) in zip(ox, oy):
    #
    #     d = math.hypot(x - agent_x, y - agent_y)
    #     angle = atan_zero_to_twopi(y - agent_y, x - agent_x)
    #     angleid = int(math.floor(angle / yawreso))
    #
    #     gridlist = precast[angleid]
    #
    #     ix = int(round((x - minx) / xyreso))
    #     iy = int(round((y - miny) / xyreso))
    #
    #     for grid in gridlist:
    #         if angle > agent_angle_deg and angle <= agent_angle_deg + agent_sight_angle_deg:
    #             if grid.d > d:
    #                 pmap[grid.ix][grid.iy] = 1

    # for x in range(xw):
    #     for y in range(yw):
    #         d = math.hypot(x - agent_x, y - agent_y)
    #         angle = atan_zero_to_twopi(y - agent_y, x - agent_x)
    #         angleid = int(math.floor(angle / yawreso))
    #         if angle > agent_angle_deg and angle <= agent_angle_deg + agent_sight_angle_deg:
    #             pmap[x][y] = 0.5
    #         else:
    #             pmap[x][y] = 1

    start_angleid = int(math.floor(agent_angle_deg / yawreso))
    end_angleid = int(math.floor((agent_angle_deg + agent_sight_angle_deg) / yawreso))
    for angleid in range(start_angleid, end_angleid + 1):
        for grid in precast[angleid]:
            if math.hypot(grid.px - agent_x, grid.py - agent_y) <= agent_sensing_range:
                pmap[grid.ix][grid.iy] = 1

    return pmap, minx, maxx, miny, maxy, xyreso


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")


def main():
    print(__file__ + " start!!")

    xyreso = 0.25  # x-y grid resolution [m]
    yawreso = np.deg2rad(10.0)  # yaw angle resolution [rad]

    for i in range(5):
        ox = (np.random.rand(4) - 0.5) * 10.0
        oy = (np.random.rand(4) - 0.5) * 10.0
        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(
            ox, oy, xyreso, yawreso)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(ox, oy, "xr")
            # plt.plot(0.0, 0.0, "ob")
            plt.plot(agent_x, agent_y, "ob")
            plt.pause(1.0)


if __name__ == '__main__':
    main()