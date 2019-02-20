from numpy import *
from math import *


def Haar2D(t, b):
    if b < 1 or (len(t) < 2 and len(t[0]) < 2):
        return t
    rows = int(math.log(len(t), 2))
    columns = int(math.log(len(t[0]), 2))
    if rows < math.log(len(t), 2):
        rows = rows + 1
    if columns < math.log(len(t[0]), 2):
        columns = columns + 1
    p = zeros((pow(2, rows), pow(2, columns)))
    for i in range(0, len(t)):
        for j in range(0, len(t[0])):
            p[i][j] = t[i][j]
    w1 = zeros((len(p), len(p)))
    w2 = zeros((len(p[0]), len(p[0])))
    for i in range(0, len(w1) / 2):
        w1[i][2 * i] = 1 / sqrt(2)
        w1[i][2 * i + 1] = 1 / sqrt(2)
        w1[i + len(w1) / 2][2 * i] = 1 / sqrt(2)
        w1[i + len(w1) / 2][2 * i + 1] = -1 / sqrt(2)
    for i in range(0, len(w2) / 2):
        w2[i][2 * i] = 1 / sqrt(2)
        w2[i][2 * i + 1] = 1 / sqrt(2)
        w2[i + len(w2) / 2][2 * i] = 1 / sqrt(2)
        w2[i + len(w2) / 2][2 * i + 1] = -1 / sqrt(2)
    print(p)
    p = dot(dot(w1, p), transpose(w2))
    print(w1[0][0])
    print(w2)
    q = Haar2D(p[0:len(p) / 2, 0:len(p) / 2], b - 1)
    for i in range(0, len(q)):
        for j in range(0, len(q)):
            p[i][j] = q[i][j]
    return p


temp = [[20, 20, 190, 190, 190, 190, 20, 20],
        [20, 20, 190, 190, 190, 190, 20, 20],
        [190, 190, 190, 190, 190, 190, 190, 190],
        [190, 190, 190, 190, 190, 190, 190, 190],
        [190, 190, 190, 190, 190, 190, 190, 190],
        [190, 190, 190, 190, 190, 190, 190, 190],
        [20, 20, 190, 190, 190, 190, 20, 20],
        [20, 20, 190, 190, 190, 190, 20, 20]]
temp1 = [[250, 23, 3, 11],
         [178, 144, 33, 24]]
print(Haar2D(temp, 3))
