import datetime
import json
import math
import random

import geopy.distance
import numpy as np

from geopy.distance import geodesic

from demo_physical.common.utils import Transform as transform
from demo_physical.common.utils import cal_JJ as cal_angle

FIRE_PITCH_ANGLE = 15
EXCURSION_SPEED = 270


class Uav:
    # dd初始化
    def __init__(self, uav_name, ddinfo_json_path, goal_position):
        self.name = uav_name
        self.a = None
        self.goal_ECEF_pos = None
        self.fly_distance = None
        self.fire_pos = None
        self.end = False
        self.final = None
        self.base_angle = None
        self.fire_director = None
        self.fire_ECEF_pos = None
        self.director = None
        self.ECEF_pos = None
        self.goal_pos = None
        self.base_direction = None
        self.json_path = ddinfo_json_path
        self.stage = None
        self.speed = None
        self.pitch_angle = None
        self.excursion_angle = None
        self.time = None
        self.pos = None

        self.drone_position_list = list()
        self.drone_position_list_ll = list()
        self.read_ddinfo_json_file(goal_position)

    def read_ddinfo_json_file(self, goal_pos):
        # 读取配置文件
        with open(self.json_path, 'r', encoding='utf-8') as f:
            load_data = json.load(f, strict=False, encoding='utf8')
        # 速度,飞行方向,倾角,偏航角,飞行时间,位置,点火位置,目标位置
        self.speed = load_data["speed"]
        self.director = None
        self.pitch_angle = load_data["pitch_angle"]
        self.excursion_angle = load_data["excursion_angle"]
        self.time = 0  # uav飞行所经历的时间
        self.pos = load_data["position"][2]
        self.fire_pos = self.pos
        self.goal_pos = [119.308165, 21.410862, 1]

        # 初始化无人机的飞行阶段,uav起始地心地固坐标,uav地心地固点火坐标,初始化目标的位置信息
        self.stage = 1
        self.ECEF_pos = transform.BLH2XYZ(self.pos)
        self.fire_ECEF_pos = self.ECEF_pos.copy()
        self.set_begin_director()
        self.fly_distance = 0
        f.close()

    # 为uav设置初始行进方向
    def set_begin_director(self):
        self.director = np.array(np.array(self.ECEF_pos) / np.linalg.norm(self.ECEF_pos))
        self.fire_director = self.director
        self.goal_ECEF_pos = transform.BLH2XYZ(self.goal_pos)
        base_director = self.goal_ECEF_pos - self.fire_ECEF_pos
        self.base_direction = np.array(self.goal_ECEF_pos) - np.array(self.fire_ECEF_pos)
        self.base_angle = cal_angle.cal_jj(base_director, self.fire_director)

    # 根据俯仰角计算uav的行进方向
    def cal_director(self):

        goal_ECEF_position = transform.BLH2XYZ(self.goal_pos)
        base_director = goal_ECEF_position - self.fire_ECEF_pos
        base_director = np.array(base_director) / np.linalg.norm(base_director)
        base_angle = cal_angle.cal_jj(base_director, self.fire_director)
        angle_1 = math.radians(base_angle - self.pitch_angle)
        angle_2 = math.radians(self.pitch_angle)
        # 计算两向量的叉乘
        normal_vector = np.cross(np.array(self.fire_director), np.array(base_director))
        normal_vector = 1 / np.linalg.norm(normal_vector) * np.array(normal_vector)
        p = np.array([np.array(base_director), np.array(self.fire_director), np.array(normal_vector)])
        y = np.array([math.cos(angle_2), math.cos(angle_1), 0])
        director = np.linalg.solve(p, y)
        self.director = director

    # 计算方位角函数
    def azimuthAngle(self):
        angle = 0.0
        x1 = self.fire_pos[0]
        y1 = self.fire_pos[1]
        x2 = self.goal_pos[0]
        y2 = self.goal_pos[1]
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return angle * 180 / math.pi

    def set_cur_position(self, position):
        self.pos = position

    # 阶段变化
    def satge_change(self):
        if self.time <= 14:
            self.stage = 1
        if 14 < self.time <= 100:
            self.stage = 2
        if self.time >= 100:
            self.stage = 3
        if self.stage == 3 and self.final is True:
            self.stage = 4

    # 设置当前dd的偏转角(实时更新)
    def set_excursion_angle(self, angle):
        self.excursion_angle = angle

    # 计算当前dd的俯仰角
    def cal_pitch_angle(self):
        if self.stage == 1:
            self.pitch_angle = self.base_angle
        if self.stage == 2:
            if self.time <= 44:
                self.pitch_angle -= 120 / 27.6
            if 100 >= self.time > 44:
                self.pitch_angle += 35 / 56
        if self.stage == 3:
            self.pitch_angle = 0
        if self.stage == 4:
            self.pitch_angle += 0.39
            if self.pitch_angle >= 5 or self.pitch_angle < 0:
                self.pitch_angle -= 42 / 5

    def cal_acclerate(self):
        if self.stage == 1:
            a = 13
        if self.stage == 2 and self.time <= 44:
            a = 57 / 25 * (-1)
        if self.stage == 2 and 100 >= self.time > 44:
            a = 140 / 50
        if self.stage == 3 or self.time == 101:
            a = 0
        if self.stage == 4:
            a = 0
        return a

    def fixed_height(self):
        blh = transform.XYZ2BLH(self.ECEF_pos)
        blh[2] = 150
        self.ECEF_pos = transform.BLH2XYZ(blh)

    # 计算当前dd在加速度影响下的速度及位移(实时更新)
    def cal_power_position(self, time):
        a = self.cal_acclerate()
        before_speed = np.array(self.director) * self.speed
        after_speed = np.array(self.director) * (self.speed + a * time)
        self.speed += a
        self.a = a
        self.speed = np.linalg.norm(after_speed)
        movement = np.array((before_speed + after_speed) / 2) * time
        return movement

    def cal_power_position_end(self, time):
        a = self.cal_acclerate()
        self.director = np.array(
            np.array(self.ECEF_pos) - np.array(transform.BLH2XYZ(self.goal_pos))) / np.linalg.norm(
            np.array(self.ECEF_pos) - np.array(transform.BLH2XYZ(self.goal_pos)))
        speed = np.array(self.director) * 285
        movement = -speed * time
        return movement

    # 每个step后更新当前无人机的实时数据 主要更新的内容有：时间、重量、位置、姿态、阶段情况、等等(实时更新)
    def drone_fly(self, gooal_pos, time, action):
        # 给当前dd时间+1
        self.time += time
        self.drone_position_list.append(np.array(self.pos))
        self.drone_position_list_ll.append(np.array([self.pos[1], self.pos[0]]).tolist())
        self.cal_pitch_angle()
        self.cal_director()
        self.goal_pos = gooal_pos
        # 阶段 1，2，4根据设定进行dd飞行
        if self.stage == 1 or self.stage == 2:
            movement = self.cal_power_position(time)
            self.ECEF_pos += movement
            self.director = np.array(
                np.array(self.ECEF_pos) - np.array(transform.BLH2XYZ(self.pos))) / np.linalg.norm(
                np.array(self.ECEF_pos) - np.array(transform.BLH2XYZ(self.pos)))
        if self.stage == 4:
            movement = self.cal_power_position_end(time)
            self.ECEF_pos += movement
        if self.stage == 3:
            angle = self.azimuthAngle()
            angle += self.excursion_angle
            angle += action / 2
            gooal_pos = geopy.distance.distance(miles=0.285 * time).destination((self.pos[1], self.pos[0]),
                                                                                bearing=angle)
            self.excursion_angle += action
            ECEF_position = transform.BLH2XYZ([gooal_pos.longitude, gooal_pos.latitude, self.pos[2]])
            director = ECEF_position - self.ECEF_pos
            director /= np.linalg.norm(director)
            # cal_angle.cal_jj(director, self.director)
            # movement = numpy.array(ECEF_position) - self.ECEF_position
            self.fly_distance += 285 * time
            self.ECEF_pos = ECEF_position
            self.pos = np.array([gooal_pos[1], gooal_pos[0], self.pos[2]])
            self.pos[2] = 150
            self.director = director
        if self.stage == 1 or self.stage == 2 or self.stage == 4:
            self.set_cur_position(transform.XYZ2BLH(self.ECEF_pos))
        # 计算dd与目标的距离
        # print("导弹位置：", self.position, "航母位置：", self.goal_position)
        if len(self.goal_pos) > 3:
            print(0)
        dd_goal_distance = geodesic((self.pos[1], self.pos[0]),
                                    (self.goal_pos[1], self.goal_pos[0])).m
        if self.stage == 1 and self.time >= 14:
            self.satge_change()
        if self.stage == 2 and self.time >= 100:
            self.satge_change()
        if dd_goal_distance <= 5000 and self.stage == 3:
            # print(self.time)
            self.final = True
            self.satge_change()
        if dd_goal_distance < 150:
            self.pos = gooal_pos
            self.end = True
        s_ = np.array([self.pos[0], self.pos[1], self.director[0], self.director[1]])
        return s_