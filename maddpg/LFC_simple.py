import numpy as np
from gym.spaces import Discrete, Box
from scipy import signal
import numpy as np


class Interconnected_grid:
    # def __init__(self):
    #     self.A = np.array([
    #         [-0.0500000000000000, 0.000135000000000000, 0, -0.000135000000000000, 0, 0, 0, 0, 0, 0, 0, 0,
    #          -0.000135000000000000,
    #          0, 0], [0, -3.33333333333333, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, -12.5000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [3.42433599241288, 0, 0, 0, -3.42433599241287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, -0.0500000000000000, 0.000145000000000000, 0, -0.000145000000000000, 0, 0, 0, 0, 0,
    #          -0.000145000000000000, 0], [0, 0, 0, 0, 0, -3.33333333333333, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, -12.5000000000000, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 3.42433599241287, 0, 0, 0, -3.42433599241287, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, -0.0500000000000000, 0.000135000000000000, 0, -0.000135000000000000, 0, 0,
    #          -0.000135000000000000], [0, 0, 0, 0, 0, 0, 0, 0, 0, -3.33333333333333, 3.33333333333333, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -12.5000000000000, 0, 0, 0, 0],
    #         [-3.42433599241287, 0, 0, 0, 0, 0, 0, 0, 3.42433599241287, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    #     self.B = np.array(
    #         [[0, 0, 0], [0, 0, 0], [12.5000000000000, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 12.5000000000000, 0],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 12.5000000000000], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    #     self.C = np.array([[370.787037037037, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 345.244252873563, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 370.787037037037, 0, 0, 1, 0, 0, 0],
    #                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    #     self.D = np.zeros((9, 3))
    #     self.sys = signal.StateSpace(self.A, self.B, self.C, self.D)
    #     # 状态矩阵，主要控制负荷的大小。
    #     # self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.01]
    #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     # 运行的次数
    #     self.time = 0
    def __init__(self):
        Tg, Tt, Tij = [0.1, 0.08, 0.09], [0.35, 0.33, 0.32], [0.015, 0.02, 0.01]
        D, H, R = [1.5, 2.1, 1.2], [21, 26, 18], [.3, .1, .1]
        self.A = np.array([
            [-D[0] / 2 / H[0], 1 / 2 / H[0], 0, -1 / 2 / H[0], 0, 0, 0, 0, 0, 0, 0, 0, -1 / 2 / H[0], 0, 0],
            [0, -1 / Tt[0], 1 / Tt[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1 / Tg[0] / R[0], 0, -1 / Tg[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2 * np.pi * Tij[0], 0, 0, 0, -2 * np.pi * Tij[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -D[1] / 2 / H[1], 1 / 2 / H[1], 0, -1 / 2 / H[1], 0, 0, 0, 0, 0, -1 / 2 / H[1], 0],
            [0, 0, 0, 0, 0, -1 / Tt[1], 1 / Tt[1], 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1 / Tg[1] / R[1], 0, -1 / Tg[1], 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2 * np.pi * Tij[1], 0, 0, 0, -2 * np.pi * Tij[1], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -D[2] / 2 / H[2], 1 / 2 / H[2], 0, -1 / 2 / H[2], 0, 0, -1 / 2 / H[2]],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / Tt[2], 1 / Tt[2], 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1 / Tg[2] / R[2], 0, -1 / Tg[2], 0, 0, 0, 0],
            [-2 * np.pi * Tij[2], 0, 0, 0, 0, 0, 0, 0, 2 * np.pi * Tij[2], 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.B = np.array(
            [[0, 0, 0], [0, 0, 0], [1 / Tg[0], 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1 / Tg[1], 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1 / Tg[2]], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.C = np.array([[D[0] + 1 / R[0], 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, D[1] + 1 / R[1], 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, D[2] + 1 / R[2], 0, 0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.D = np.zeros((9, 3))
        self.sys = signal.StateSpace(self.A, self.B, self.C, self.D)
        # 状态矩阵，主要控制负荷的大小。
        self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.01]
        from numpy import genfromtxt
        self.发电数据 = genfromtxt(r'负荷波动数据（用于训练）.csv', delimiter=',', skip_header=1)[:, 1:]

    def reset(self):
        # 重置状态矩阵,进入下一个回合。
        self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.time = 0

    def 改变负荷(self):
        # if self.time == 50:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01]
        # if self.time == 75:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0.01]
        # if self.time == 100:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.03, 0.01]
        # if self.time == 125:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01, 0.03, 0.01]
        # if self.time == 150:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01, 0.03, -0.02]
        # if self.time == 175:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01, 0.02, -0.02]
        # if self.time == 200:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01, 0.02, -0.02]
        # if self.time == 225:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, -0.01]
        # if self.time == 275:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, -0.01]

        # if self.time == 25:
        #     self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.02, 0.01]

        self.x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.发电数据[self.time][0], self.发电数据[self.time][1], self.发电数据[self.time][2]]

    def step(self, u, time=4):
        self.改变负荷()
        self.time += 1
        t = np.arange(0, time, 0.5)  # 每一个回合的时间，4秒控制一次发电功率。
        u = np.reshape(np.array(u), (3, 1))
        u = np.tile(u, len(t)).T  # 这里应该是发电功率
        [T, y, x] = signal.lsim(self.sys, u, t, self.x0)
        self.x0 = x[-1, :]  # 更新状态向量
        return y[-1, 0:3], y[-1, 3:6], y[-1, 6:9], np.mean(np.diff(y[:, 0:3], axis=0), axis=0), \
               self.x0[1:10:4], self.x0[2:11:4], self.x0[-3:], \
               np.mean(np.diff(y[:, 3:6], axis=0), axis=0), np.mean(np.diff(y[:, 6:9], axis=0), axis=0), \
               np.mean(np.diff(x[:, 1:10:4], axis=0), axis=0), np.mean(np.diff(x[:, 2:11:4], axis=0), axis=0)
        # ACE, F, Ptie, 导ACE
        # Pt, Xg, PL
        # 导F，导Ptie
        # 导Pt, 导Xg


class LFC:
    def __init__(self):
        self.n = 3  # 智能体数量
        self.Interconnected_grid = Interconnected_grid()
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32) for _ in
                                  range(3)]  # 状态空间的数量
        self.action_space = [Discrete(51), Discrete(51), Discrete(51)]  # 动作空间
        self.obs_shape_n = [11, 11, 11]  # 状态空间的数量
        self.act_shape_n = [51, 51, 51]  # 动作空间的数量

        self.区域1发电指令, self.区域2发电指令, self.区域3发电指令 = 0, 0, 0
        self.区域1发电指令_上, self.区域2发电指令_上, self.区域3发电指令_上 = 0, 0, 0

    def reset(self):  # 重置环境
        self.区域1发电指令, self.区域2发电指令, self.区域3发电指令 = 0, 0, 0
        self.区域1发电指令_上, self.区域2发电指令_上, self.区域3发电指令_上 = 0, 0, 0
        self.Interconnected_grid.reset()
        [ACE, F, Ptie, DACE, Pt, Xg, PL, DF, DPtie, DPt, DXg] = self.Interconnected_grid.step(
            u=[self.区域1发电指令, self.区域2发电指令, self.区域3发电指令], time=1)
        发电指令 = [self.区域1发电指令, self.区域2发电指令, self.区域3发电指令]
        当前状态 = [np.array([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]) for
                A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 in
                zip(ACE, F, DACE, Ptie, Pt, Xg, 发电指令, DF, DPtie, DPt, DXg)]
        return 当前状态

    def step(self, action_n):
        action_n = [np.argmax(i) for i in action_n]
        # diction = np.linspace(-0.001, 0.001, 51).tolist()
        diction = np.linspace(-0.003, 0.003, 51).tolist()
        发电增幅1, 发电增幅2, 发电增幅3 = [diction[i] for i in action_n]
        # self.区域1发电指令, self.区域2发电指令, self.区域3发电指令 = 发电增幅1, 发电增幅2, 发电增幅3
        self.区域1发电指令, self.区域2发电指令, self.区域3发电指令 = [x + y for x, y in zip([发电增幅1, 发电增幅2, 发电增幅3],
                                                                          [self.区域1发电指令, self.区域2发电指令, self.区域3发电指令])]
        [ACE, F, Ptie, DACE, Pt, Xg, PL, DF, DPtie, DPt, DXg] = self.Interconnected_grid.step(
            u=[self.区域1发电指令, self.区域2发电指令, self.区域3发电指令])
        发电指令 = [self.区域1发电指令, self.区域2发电指令, self.区域3发电指令]
        下一状态 = [np.array([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]) for
                A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 in
                zip(ACE, F, DACE, Ptie, Pt, Xg, 发电指令, DF, DPtie, DPt, DXg)]
        是否完成 = self.是否完成仿真(ACE)
        奖励 = self.奖励函数(ACE, DACE=DACE, 发电增幅=[发电增幅1, 发电增幅2, 发电增幅3], 完成=是否完成, 发电功率=发电指令)
        return 下一状态, 奖励, 是否完成, None

    def close(self):
        pass

    def 奖励函数(self, ACE, DACE, 发电增幅, 完成, 发电功率):
        # cost = np.square(ACE * 100) + 0.01 * np.square(np.array(发电增幅) * 10)
        #
        # cost = np.array([10 if i == True else cost[index] for index, i in enumerate(完成)])
        # 奖励 = -1 * cost

        # 奖励 = -np.square(ACE) * 50 * 100
        # 奖励 = -np.square(ACE * 50)
        # 奖励 = -np.array([np.max(np.square(ACE)).tolist()] * 3) * 50
        # 奖励 = -np.sum(np.abs([0.01 - 发电功率[0], 0.02 - 发电功率[1], 0.01 - 发电功率[2]])).repeat(3) * 10
        # 奖励 = -np.sum(ACE ** 2).repeat(3) * 50

        # ------------------------------考虑发电指令变化的奖励------------------------------
        奖励 = -np.square(ACE) * 50
        # 奖励 = -np.square(ACE) * 50 - np.abs(np.array([self.区域1发电指令 - self.区域1发电指令_上,
        #                                             self.区域2发电指令 - self.区域2发电指令_上,
        #                                             self.区域3发电指令 - self.区域3发电指令_上])) * .1
        # 奖励 = -np.sum(ACE ** 2).repeat(3) * 50 - np.abs(np.array([self.区域1发电指令 - self.区域1发电指令_上,
        #                                                          self.区域2发电指令 - self.区域2发电指令_上,
        #                                                          self.区域3发电指令 - self.区域3发电指令_上])) * .1

        奖励 = 奖励 * 100
        # 奖励 = np.array([-10 if i == True else 奖励[index] for index, i in enumerate(完成)])
        self.区域1发电指令_上, self.区域2发电指令_上, self.区域3发电指令_上 = self.区域1发电指令, self.区域2发电指令, self.区域3发电指令
        return 奖励

    def 是否完成仿真(self, ACE):
        完成 = [True if np.abs(i) > 0.08 else False for i in ACE]
        return 完成
