import numpy as np
from matplotlib import pyplot as plt


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
    # circle(dd[0], dd[1], 0.01)
    circle(jc[0][0], jc[0][1], 0.001)
    for i in leida:
        circle(i[0], i[1], 100 / 111 / len(leida))
    for i in jc[1:]:
        circle(i[0], i[1], 10.5 / 111)
    plt.pause(0.01)
    plt.show()


if __name__ == '__main__':
    pic([1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4])
