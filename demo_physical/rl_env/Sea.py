import json
import math
import queue
import random

import numpy as np
from geopy.distance import geodesic

from demo_physical.common.utils import Transform, cal_JJ

JC_NUM = 12
RADAR_MIN_LON = 116.05111
RADAR_MAX_LON = 120.94889
PER_LONLAT_KM = 111.32


class Sea:
    def __init__(self, json_file):
        self.jc_num = JC_NUM
        self.jc_attack_distance = np.empty((JC_NUM, 2), dtype=np.int)
        self.jc_weapon_state = np.empty((JC_NUM, 2), dtype=np.int)
        self.jc_states = []
        self.jc_position_list = []  # 舰船位置
        self.time = 0  # 步数
        self.zyjt_list = None
        self.xyj_list = None
        self.radar_pos_list = None
        self.radar_radius = None
        self.director = None
        self.hm_pos = None
        self.hwj_pos_list = None
        self.qzj_pos_list = None
        self.jc_speed = 10
        self.jc_cur_pos_list = None
        self.json_path = json_file
        self.json_data = None
        self.radar_nums = None
        self.read_json_file()

    # 读取json格式文件
    def read_json_file(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            load_data = json.load(f, strict=False, encoding='utf8')
        self.hm_pos = load_data[0]["list"][0]["points"]
        self.qzj_pos_list = list()
        self.jc_cur_pos_list = list()
        self.radar_pos_list = list()
        self.hwj_pos_list = list()
        self.zyjt_list = list()
        self.xyj_list = list()
        self.qzj_pos_list.append(load_data[1]["list"][0]["point"])
        self.qzj_pos_list.append(load_data[1]["list"][1]["point"])
        self.qzj_pos_list.append(load_data[1]["list"][2]["point"])
        self.qzj_pos_list.append(load_data[1]["list"][3]["point"])
        self.qzj_pos_list.append(load_data[1]["list"][4]["point"])
        self.qzj_pos_list.append(load_data[1]["list"][5]["point"])
        self.hwj_pos_list.append(load_data[2]["list"][0]["point"])
        self.zyjt_list.append(load_data[3]["list"][0]["point"])
        self.zyjt_list.append(load_data[3]["list"][1]["point"])
        self.xyj_list.append(load_data[4]["list"][0]["point"])
        self.xyj_list.append(load_data[4]["list"][1]["point"])
        self.jc_cur_pos_list.append(np.array(self.hm_pos[0]))

        self.jc_attack_distance[0, 0] = load_data[0]["list"][0]["long_attack_distance"]
        self.jc_weapon_state[0, 0] = 10
        self.jc_attack_distance[0, 1] = load_data[0]["list"][0]["attack_distance"]
        self.jc_weapon_state[0, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[0]).tolist())

        self.jc_attack_distance[1, 0] = load_data[1]["list"][0]["long_attack_distance"]
        self.jc_weapon_state[1, 0] = 10
        self.jc_attack_distance[1, 1] = load_data[1]["list"][0]["attack_distance"]
        self.jc_weapon_state[1, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[1]).tolist())

        self.jc_attack_distance[2, 0] = load_data[1]["list"][1]["long_attack_distance"]
        self.jc_weapon_state[2, 0] = 10
        self.jc_attack_distance[2, 1] = load_data[1]["list"][1]["attack_distance"]
        self.jc_weapon_state[2, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[2]).tolist())

        self.jc_attack_distance[3, 0] = load_data[1]["list"][2]["long_attack_distance"]
        self.jc_weapon_state[3, 0] = 10
        self.jc_attack_distance[3, 1] = load_data[1]["list"][2]["attack_distance"]
        self.jc_weapon_state[3, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[5]).tolist())

        self.jc_attack_distance[4, 0] = load_data[1]["list"][5]["long_attack_distance"]
        self.jc_weapon_state[4, 0] = 10
        self.jc_attack_distance[4, 1] = load_data[1]["list"][5]["attack_distance"]
        self.jc_weapon_state[4, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.hwj_pos_list[0]).tolist())

        self.jc_attack_distance[5, 0] = load_data[2]["list"][0]["long_attack_distance"]
        self.jc_weapon_state[5, 0] = 10
        self.jc_attack_distance[5, 1] = load_data[2]["list"][0]["attack_distance"]
        self.jc_weapon_state[5, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.zyjt_list[0]).tolist())

        self.jc_attack_distance[6, 0] = load_data[3]["list"][0]["long_attack_distance"]
        self.jc_weapon_state[6, 0] = 0
        self.jc_attack_distance[6, 1] = load_data[3]["list"][0]["attack_distance"]
        self.jc_weapon_state[6, 1] = 0
        self.jc_cur_pos_list.append(np.array(self.zyjt_list[1]).tolist())

        self.jc_attack_distance[7, 0] = load_data[3]["list"][1]["long_attack_distance"]
        self.jc_weapon_state[7, 0] = 0
        self.jc_attack_distance[7, 1] = load_data[3]["list"][1]["attack_distance"]
        self.jc_weapon_state[7, 1] = 0
        self.jc_cur_pos_list.append(np.array(self.xyj_list[0]).tolist())

        self.jc_attack_distance[8, 0] = load_data[4]["list"][0]["long_attack_distance"]
        self.jc_weapon_state[8, 0] = 10
        self.jc_attack_distance[8, 1] = load_data[4]["list"][0]["attack_distance"]
        self.jc_weapon_state[8, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.xyj_list[1]).tolist())

        self.jc_attack_distance[9, 0] = load_data[4]["list"][1]["long_attack_distance"]
        self.jc_weapon_state[9, 0] = 10
        self.jc_attack_distance[9, 1] = load_data[4]["list"][1]["attack_distance"]
        self.jc_weapon_state[9, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[3]).tolist())

        self.jc_attack_distance[10, 0] = load_data[1]["list"][3]["long_attack_distance"]
        self.jc_weapon_state[10, 0] = 10
        self.jc_attack_distance[10, 1] = load_data[1]["list"][3]["attack_distance"]
        self.jc_weapon_state[10, 1] = 10
        self.jc_cur_pos_list.append(np.array(self.qzj_pos_list[4]).tolist())

        self.jc_attack_distance[11, 0] = load_data[1]["list"][3]["long_attack_distance"]
        self.jc_weapon_state[11, 0] = 10
        self.jc_attack_distance[11, 1] = load_data[1]["list"][3]["attack_distance"]
        self.jc_weapon_state[11, 1] = 10

        # 初始化雷达数量,位置
        self.radar_nums = 3
        leida_radiu = 80 * 1000

        MIN_LONGTITUDE = 22.254 + leida_radiu / (PER_LONLAT_KM * 1000)
        MAX_LONGTITUDE = 24.745 - leida_radiu / (PER_LONLAT_KM * 1000)
        self.radar_radius = leida_radiu

        for index in range(self.radar_nums):
            leida_pos = list()
            leida_pos.append(random.uniform(RADAR_MIN_LON, RADAR_MAX_LON))
            leida_pos.append(random.uniform(MIN_LONGTITUDE, MAX_LONGTITUDE))
            self.radar_pos_list.append(leida_pos)

        self.director = np.array(
            Transform.BLH2XYZ(self.hm_pos[1]) - Transform.BLH2XYZ(self.hm_pos[0]))
        self.director = np.array(self.director) / np.linalg.norm(self.director)
        f.close()

    def update_jc_position(self, goal_index=None, time=1):
        self.time += time
        self.jc_position_list.append(self.jc_cur_pos_list)
        for index in range(len(self.jc_cur_pos_list)):
            position = self.jc_cur_pos_list[index]
            position_ECEF = Transform.BLH2XYZ(position)
            position_ECEF += np.array(self.director) * self.jc_speed
            position = Transform.XYZ2BLH(position_ECEF)
            position[2] = 1
            self.jc_cur_pos_list[index] = np.array(position)
        return self.jc_cur_pos_list
