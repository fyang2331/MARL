import json

import gym
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt

from demo_physical.rl_env.Drone import Uav
from demo_physical.rl_env.Sea import Sea


def circle(x, y, r):
    # 点的横坐标为a
    u = np.linspace(x - r, x + r, 300)
    y1 = np.sqrt(r ** 2 - (u - x) ** 2) + y
    y2 = -np.sqrt(r ** 2 - (u - x) ** 2) + y

    plt.plot(u, y1, c='k')
    plt.plot(u, y2, c='k')


def pic(dd, leida, jc, hm):
    plt.ion()
    plt.cla()
    plt.xlim(117.493, 120.493)
    plt.ylim(20.684, 24.684)
    circle(dd[0], dd[1], 0.01)
    circle(jc[0][0], jc[0][1], 0.001)
    for i in leida:
        circle(i[0], i[1], 100 / 111 / len(leida))
    for i in jc[1:]:
        circle(i[0], i[1], 10.5 / 111)
    plt.pause(0.01)
    plt.show()


SEA_MAX_LAT = 24.746
SEA_MIN_LAT = 20.254
SEA_MAX_LON = 120.949
SEA_MIN_LON = 116.051
JC_INDEX_MAP = {"CVN-76": "0", "DDG-87": "1", "DDG-85": "2", "DDG-65": "3", "DDG-51": "4", "FFG-51": "5",
                "AOE-8": "6", "AOE-7": "7", "CG-63": "8", "CG-67": "9", "DDG-89": "10", "DDG-93": "11"}
INDEX_JC_MAP = {"0": "CVN-76", "1": "DDG-87", "2": "DDG-85", "3": "DDG-65", "4": "DDG-51", "5": "FFG-51",
                "6": "AOE-8", "7": "AOE-7", "8": "CG-63", "9": "CG-67", "10": "DDG-89", "11": "DDG-93"}
STATE_SPACE_SIZE = 50


class SeaBattleEnv(gym.Env):
    def __init__(self, jc_json_path, drone_json_path, agent_num):
        self.sea = None

        # 智能体数量、智能体实例化容器、智能体步数、智能体阶段、智能体状态、智能体名称
        self.drone_num = agent_num
        self.drones = None
        self.drones_time = None
        self.drones_stage = None
        self.drones_isEnd_state = np.array([])
        self.drone_goal_names = None

        self.before_distance = None
        self.after_distance = None

        self.drone_radar_dis = None

        self.jc_json_path = jc_json_path
        self.drone_info_json_path = drone_json_path

        self.action_space = gym.spaces.Box(
            low=-30,
            high=30,
            shape=(1,),
            dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(50, 77),
            dtype=np.float32)
        self.build_env()
        self.total_r = 0

    def build_env(self):
        return

    def reset(self):
        # 初始化总奖励值
        self.total_r = 0
        # 初始化海域,初始化目标
        self.sea = Sea(self.jc_json_path)
        self.drone_goal_names = np.full((self.drone_num,), "CVN-76")

        # 初始化无人机集群
        self.drones = []
        for i in range(0, self.drone_num):
            drone_name = "drone-" + str(i)
            self.drones_isEnd_state = np.append(self.drones_isEnd_state, False)
            self.drones.append(Uav(drone_name, self.drone_info_json_path, self.drone_goal_names[i]))

        # 初始化无人机集群中各无人机步数
        self.drones_time = np.zeros((self.drone_num,))
        self.drones_stage = np.ones((self.drone_num,))

        # 无人机集群起飞阶段,前三十架无人机先起飞,剩余无人机过5s后起飞
        drones_observation_array = np.array([])
        drones_info_observation = self.step_for_1_2_4(before_num=30, after_num=20)
        jc_position_list = self.sea.jc_cur_pos_list
        self.before_distance = self.cal_drones_jc_dis(jc_position_list)
        for i in range(self.drone_num):
            drone = self.drones[i]
            # 初始化无人机信息
            drone_observation = drones_info_observation[i]

            # 舰船位置,初始化无人机和舰船的距离
            drone_jc_dis = self.cal_drone_jc_dis(jc_position_list, drone)
            drone_observation = np.append(drone_observation, drone_jc_dis)

            # 将12艘舰船的位置放入状态空间
            for i in range(len(jc_position_list)):
                drone_observation = np.append(drone_observation, self.sea.jc_cur_pos_list[i][0])
                drone_observation = np.append(drone_observation, self.sea.jc_cur_pos_list[i][1])

            # 将各舰船的武器系统的抵御范围放入状态空间
            drone_observation = np.append(drone_observation, self.sea.jc_attack_distance)
            drone_observation = self.radar_info_drones_dis(drone_observation, drone)
            drones_observation_array = np.append(drones_observation_array, drone_observation)

        return drones_observation_array

    def radar_info_drones_dis(self, state, drone):
        self.drone_radar_dis = np.array([])
        for index1 in range(self.sea.radar_nums):
            state = np.append(state, self.sea.radar_pos_list[index1][0:2])
            state = np.append(state, 80 * 1000)
            drone_radar_dis = geodesic(
                (self.sea.radar_pos_list[index1][1], self.sea.radar_pos_list[index1][0]),
                (drone.pos[1], drone.pos[0])).m
            self.drone_radar_dis = np.append(self.drone_radar_dis, drone_radar_dis)
            state = np.append(state, drone_radar_dis)
        return state

    def cal_drone_jc_dis(self, jc_position_list, drone):
        drone_jc_distance = np.empty((self.sea.jc_num,), dtype=np.float)  #
        for index in range(len(jc_position_list)):
            drone_jc_distance[index] = geodesic((jc_position_list[index][1], jc_position_list[index][0]),
                                                (drone.pos[1], drone.pos[0])).m
        return drone_jc_distance

    def step_for_1_2_4(self, before_num=30, after_num=20):
        # 无人机飞行,前三十架飞行渡过1,2阶段,后二十架飞行距离2阶段结束差5步
        observation = np.empty((self.drone_num, 5))
        for i in range(self.drone_num):
            if i <= before_num:
                while self.drones[i].stage == 1 or self.drones[i].stage == 2:
                    self.sea.update_jc_position(time=1)
                    self.drone_fly_1_2(i)
            else:
                while self.drones[i].time <= self.drones[0].time - 5:
                    self.sea.update_jc_position(time=1)
                    self.drone_fly_1_2(i)
        for i in range(1, self.drone_num):
            while self.drones[i].stage == 4 and self.drones[i].end is not True:
                self.sea.update_jc_position(time=1)
                goal_position = self.sea.jc_cur_position_list[int(JC_INDEX_MAP.get(self.drone_goal_names[i]))]
                observation = self.drones[i].drone_fly(goal_position, 1, action=None)
        for i in range(self.drone_num):
            observation[i] = np.append(self.drones[i].pos[0:2], self.drones[i].director)
            self.drones_time = self.drones[i].time
        return observation

    def drone_fly_1_2(self, i):
        goal_name = self.drone_goal_names[i]
        self.drones[i].drone_fly(self.sea.jc_cur_pos_list[int(JC_INDEX_MAP.get(goal_name))], 1, action=None)

    def step(self, actions):
        # 初始化奖励,游戏状态,状态空间初始化
        reward = 0
        Done = False
        s_ = np.array([])
        all_drones_failed_flag = True

        # 根据策略传来的动作计算实际的偏转角动作
        true_action = np.empty((50,), dtype=np.float)
        for i in range(len(actions)):
            true_action[i] = np.clip(actions[i], -1, 1) * 50

        # 航母编队运动
        self.sea.update_jc_position(time=1)

        # 无人机集群进行飞行、奖励结算
        for i in range(self.drone_num):
            drone_s_ = np.array([])
            drone = self.drones[i]
            drone_goal_dis_before = self.before_distance[i, int(JC_INDEX_MAP.get(self.drone_goal_names[i]))]
            if drone.time >= 100:
                goal_name = self.drone_goal_names[i]
                goal_pos = self.sea.jc_cur_pos_list[int(JC_INDEX_MAP.get(goal_name))]
                drone.drone_fly(gooal_pos=goal_pos, time=1, action=true_action[i])
            else:
                self.drone_fly_1_2(i)
            if drone.end:
                drone_s_ = np.zeros((77,))
                s_ = np.append(s_, drone_s_)
                continue
            drone_s_ = np.append(drone_s_, self.drones[i].pos[0:2])
            drone_s_ = np.append(drone_s_, self.drones[i].director)

            # 舰船位置,无人机与舰船距离
            for j in range(len(self.sea.jc_cur_pos_list)):
                drone_s_ = np.append(drone_s_, self.sea.jc_cur_pos_list[j][0])
                drone_s_ = np.append(drone_s_, self.sea.jc_cur_pos_list[j][1])
            drone_after_distance = self.cal_drone_jc_dis(self.sea.jc_cur_pos_list, drone)
            drone_s_ = np.append(drone_s_, drone_after_distance)

            # 舰船武器系统数据
            drone_s_ = np.append(drone_s_, self.sea.jc_attack_distance)

            # 雷达位置,范围,与无人机距离
            drone_s_ = self.radar_info_drones_dis(drone_s_, drone)
            s_ = np.append(s_, drone_s_)

            drone_jc_dis = self.cal_drone_jc_dis(self.sea.jc_cur_pos_list, drone)
            drone_goal_dis_after = drone_jc_dis[int(JC_INDEX_MAP.get(self.drone_goal_names[i]))]
            if drone_jc_dis[0] < 5000:
                Done = True
                reward += 100
                break
            break_jc_weapon_sys_flag, break_jc_weapon_sys_r = self.break_jc_weapon_sys(drone_jc_dis, i)
            if break_jc_weapon_sys_flag:
                reward += break_jc_weapon_sys_r
            radar_spot_drone_flag, radar_spot_drone_r = self.radar_spot_drone(i)
            if radar_spot_drone_flag:
                reward += radar_spot_drone_r
            # 模拟dd被销毁的过程
            if self.drones[i].pos[0] < 116.05111 or self.drones[i].pos[0] > \
                    120.94889 or self.drones[i].pos[1] > 24.74578 or \
                    self.drones[i].pos[1] < 20.25422:
                reward -= 10
                drone.end = True
                self.drones_isEnd_state[i] = True
                # print("编号为:" + str(i) + "飞机出界")
            if self.drones[i].time > 2000:
                reward -= 10
                drone.end = True
                self.drones_isEnd_state[i] = True
                # print("编号为:" + str(i) + "飞机超出飞行步数")
            if not drone.end:
                reward += (drone_goal_dis_before - drone_goal_dis_after) / (1000 * 1000)
                all_drones_failed_flag = False
        if all_drones_failed_flag and Done is not True:
            Done = True
            reward -= 100
        self.before_distance = np.copy(self.after_distance)
        # 舰船与无人机集群间的距离计算
        self.before_distance = self.cal_drones_jc_dis(self.sea.jc_cur_pos_list)
        self.total_r += reward
        return s_, reward, Done, {}

    def break_jc_weapon_sys(self, drone_jc_dis, drone_index):
        break_reward = None
        for i in range(len(drone_jc_dis)):
            if i == 0:
                break_reward = 5
            else:
                break_reward = 2
            if (drone_jc_dis[i] < self.sea.jc_attack_distance[i, 0] and self.sea.jc_weapon_state[
                i, 0] > 0) or (
                    drone_jc_dis[i] < self.sea.jc_attack_distance[i, 1] and self.sea.jc_weapon_state[
                i, 1] > 0):
                self.drones[drone_index].end = True
                self.drones_isEnd_state[drone_index] = True
                print("编号为:" + str(drone_index) + "飞机被舰船击毁")
                self.sea.jc_weapon_state[0, 0] -= 1
                return True, break_reward
        return False, 0

    def radar_spot_drone(self, drone_index):
        flag = False
        spot_reward = 0
        for i in range(self.sea.radar_nums):
            if self.drone_radar_dis[i] < self.sea.radar_radius:
                spot_reward = -5
                self.drones[drone_index].end = True
                flag = True
                self.drones_isEnd_state[drone_index] = True
                print("编号为:" + str(drone_index) + "飞机被雷达监测到")
        return flag, spot_reward

    def render(self):
        return

    def cal_drones_jc_dis(self, jc_position_list):
        drones_jc_distance = np.empty((50, 12))
        for i in range(self.drone_num):
            drone_jc_distance = np.array([])
            for index in range(len(jc_position_list)):
                drone_jc_distance = np.append(drone_jc_distance,
                                              geodesic((jc_position_list[index][1], jc_position_list[index][0]),
                                                       (self.drones[i].pos[1], self.drones[i].pos[0])).m)
            drones_jc_distance[i] = drone_jc_distance
        return drones_jc_distance


def writeDataIntoJson(jsonPath: str, data):
    with open(jsonPath, "w", encoding='utf-8') as f:
        # json.dump(dict_, f)  # 写为一行
        json.dump(data, f, indent=2, sort_keys=False,
                  ensure_ascii=False)  # 写为多行


if __name__ == '__main__':
    env = SeaBattleEnv("../json_file/env.json", "../json_file/dd_info.json", 50)
    for t in range(1):
        env.reset()
        step = 0
        while True:
            env.render()
            a = np.zeros((50,))
            s, r, done, info = env.step(a)
            step += 1
            print(r)
            print(env.drones_isEnd_state)
            if done:
                break
