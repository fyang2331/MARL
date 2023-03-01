import numpy as np


def cal_jj(vecter1, vecter2):
    if np.linalg.norm(vecter1) == 0 or np.linalg.norm(vecter2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(vecter1, vecter2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(vecter1) * np.linalg.norm(vecter2)))
        angle = np.degrees(arccos)
        return angle
    return 0

