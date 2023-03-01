import copy
import json
import os


class SuccessExcuter:
    dd_success_result_0 = set()
    dd_success_result_1 = set()
    dd_success_result_2 = set()
    dd_success_result_3 = set()
    dd_success_result_4 = set()
    jc_index_mapper = {"0": "CVN-76", "1": "DDG-87", "2": "DDG-85", "3": "DDG-65", "4": "DDG-51", "5": "FFG-51",
                       "6": "AOE-8", "7": "AOE-7", "8": "CG-63", "9": "CG-67", "10": "DDG-89", "11": "DDG-93"}

    def __init__(self):
        self.result_path = "/home/lou/Documents/Yf_project/projects/demo_half_chenge_goal/demo_physical/success_result/"
        self.file_index = 1
        self.env_json_format = None
        self.dd0 = str([118.5, 24.7, 0])
        self.dd1 = str([118.55, 24.75, 0])
        self.dd2 = str([118.6, 24.8, 0])
        self.dd3 = str([118.65, 24.85, 0])
        self.dd4 = str([118.7, 24.9, 0])
        self.file_content = None
        self.read_txt_file()
        self.read_json_file()
        self.classify()

    def read_txt_file(self):        # self.write_to_env_json()
        with open("/home/lou/Documents/Yf_project/projects/demo_half_chenge_goal/demo_physical/result.txt", "r") as f:
            lines = f.readlines()
        self.file_content = lines
        f.close()

    def classify(self):
        for line in self.file_content:
            line.replace("\\n", "")
            if self.dd0 in line:
                self.dd_success_result_0.add(line)
            if self.dd1 in line:
                self.dd_success_result_1.add(line)
            if self.dd2 in line:
                self.dd_success_result_2.add(line)
            if self.dd3 in line:
                self.dd_success_result_3.add(line)
            if self.dd4 in line:
                self.dd_success_result_4.add(line)

    def read_json_file(self):
        f = open("/home/lou/Documents/Yf_project/projects/demo_half_chenge_goal/demo_physical/json_file/demo.json",
                 encoding="utf-8")
        self.env_json_format = json.load(f)
        f.close()

    def write_to_env_json(self):
        # dd_success_result = [self.dd_success_result_0, self.dd_success_result_1, self.dd_success_result_2,
        #                      self.dd_success_result_3, self.dd_success_result_4]
        dd_success_result = [self.dd_success_result_2]
        for results in dd_success_result:
            for result in results:
                self.transform_to_json(result)

    def transform_to_json(self, content):
        content = content[0:len(content) - 1]
        env_json = copy.deepcopy(self.env_json_format)
        for index in range(len(env_json)):
            if env_json[index]["type"] == "dd":
                for dd_dict in env_json[index]["list"]:
                    if self.fire_position_in_dict(dd_dict, content):
                        dd_dict["target"] = self.jc_index_mapper.get(str(content).split("]")[1].split(" ")[1])
                        dd_dict["change_goal_name"] = self.jc_index_mapper.get(str(content).split("]")[1].split(" ")[2])
                        self.write_result_to_json(env_json)

    def fire_position_in_dict(self, dd_dict, dd_position):
        if str(dd_dict["points"][0]) in dd_position:
            return True
        return False

    def write_result_to_txt(self):
        with open("result.txt", "a") as f:
            f.write(result + "\n")

    def write_result_to_json(self, env_json):
        res = []
        for l in env_json:
            res.append(l)
        file_path = self.result_path + str(copy.deepcopy(self.file_index)) + ".json"
        self.file_index += 1
        with open(file_path, "w") as outfile:
            json.dump(res, outfile)
        outfile.close()


if __name__ == '__main__':
    path = "/home/lou/Documents/Yf_project/projects/demo_half_chenge_goal/demo_physical/success_result/"
    if not os.path.exists(path):
        os.mkdir(path)
    s = SuccessExcuter()
    print()
