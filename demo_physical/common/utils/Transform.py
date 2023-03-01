"""
CoordTransform模块用以实现大地坐标系与参心空间直角坐标系之间的坐标转换.

模块主要包括以下函数
BLH2XYZ
       该函数可以把某点的大地坐标(B, L, H)转换为空间直角坐标(X, Y, Z)
XYZ2BLH
       该函数可以把某点的空间直角坐标(X, Y, Z)转换为大地坐标（B, L, H)
rad2angle
       该函数把弧度转换为角度
angle2rad
       该函数把角度转换为弧度

"""
import math as mt

import numpy as np


def BLH2XYZ(blh):
    """
     该函数把某点的大地坐标(B, L, H)转换为空间直角坐标（X, Y, Z).
    :param B:  大地纬度, 角度制, 规定南纬为负，范围为[-90, 90]
    :param L:  大地经度, 角度制, 规定西经为负, 范围为[-180, 180]
    :param H:  海拔，大地高, 单位 m
    :param a:  地球长半轴，即赤道半径，单位 m
    :param b:  地球短半轴，即大地坐标系原点到两级的距离, 单位 m
    :return:   X, Y, Z, 空间直角坐标, 单位 m
    """
    B = blh[1]
    L = blh[0]
    H = blh[2]
    a = 6378137.0  # 参考椭球的长半轴, 单位 m
    b = 6356752.31414  # 参考椭球的短半轴, 单位 m

    sqrt = mt.sqrt
    sin = mt.sin
    cos = mt.cos
    B = angle2rad(B)  # 角度转为弧度
    L = angle2rad(L)  # 角度转为弧度

    e = sqrt((a ** 2 - b ** 2) / (a ** 2))  # 参考椭球的第一偏心率
    N = a / sqrt(1 - e * e * sin(B) * sin(B))  # 卯酉圈半径, 单位 m

    X = (N + H) * cos(B) * cos(L)
    Y = (N + H) * cos(B) * sin(L)
    Z = (N * (1 - e ** 2) + H) * sin(B)
    return np.array([X, Y, Z])  # 返回空间直角坐标(X, Y, Z)


def XYZ2BLH(xyz):
    """
    该函数实现把某点在参心空间直角坐标系下的坐标（X, Y, Z)转为大地坐标（B, L, H).
    :param X:  X方向坐标，单位 m
    :param Y:  Y方向坐标, 单位 m
    :param Z:  Z方向坐标, 单位 m
    :param a: 地球长半轴，即赤道半径，单位 m
    :param b: 地球短半轴，即大地坐标系原点到两级的距离, 单位 m
    :return:  B, L, H, 大地纬度、经度、海拔高度 (m)
    """
    X = xyz[0]
    Y = xyz[1]
    Z = xyz[2]
    a = 6378137.0  # 参考椭球的长半轴, 单位 m
    b = 6356752.31414  # 参考椭球的短半轴, 单位 m
    sqrt = mt.sqrt
    sin = mt.sin
    cos = mt.cos
    atan = mt.atan

    e = sqrt((a ** 2 - b ** 2) / (a ** 2))

    if X == 0 and Y > 0:
        L = 90
    elif X == 0 and Y < 0:
        L = -90
    elif X < 0 and Y >= 0:
        L = atan(Y / X)
        L = rad2angle(L)
        L = L + 180
    elif X < 0 and Y <= 0:
        L = atan(Y / X)
        L = rad2angle(L)
        L = L - 180
    else:
        L = atan(Y / X)
        L = rad2angle(L)

    b0 = atan(Z / sqrt(X ** 2 + Y ** 2))
    N_temp = a / sqrt((1 - e * e * sin(b0) * sin(b0)))
    b1 = atan((Z + N_temp * e * e * sin(b0)) / sqrt(X ** 2 + Y ** 2))

    while abs(b0 - b1) > 1e-7:
        b0 = b1
        N_temp = a / sqrt((1 - e * e * sin(b0) * sin(b0)))
        b1 = atan((Z + N_temp * e * e * sin(b0)) / sqrt(X ** 2 + Y ** 2))

    B = b1
    N = a / sqrt((1 - e * e * sin(B) * sin(B)))
    H = sqrt(X ** 2 + Y ** 2) / cos(B) - N
    B = rad2angle(B)
    return np.array([L, B, H])  # 返回大地纬度B、经度L、海拔高度H (m)


def rad2angle(r):
    """
    该函数可以实现弧度到角度的转换.
    :param r:  弧度
    :return:  a, 对应的角度
    """
    a = r * 180.0 / mt.pi
    return a


def angle2rad(a):
    """
    该函数可以实现角度到弧度的转换.
    :param a:  角度
    :return:  r, 对应的弧度
    """
    r = a * mt.pi / 180.0
    return r


if __name__ == '__main__':
    blh = XYZ2BLH([-2858318.057550883, 5197945.198252973, 2335647.5658845063])
    print(blh)
