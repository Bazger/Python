from numpy import math


def Haar(t, b):
    if len(t) < 2 or b < 1:
        return t
    if b > math.log(len(t), 2):
        b = math.log(len(t), 2)
    a = []
    c = []
    for i in range(0, len(t), 2):
        a.append((t[i] + t[i + 1]) / math.sqrt(2))
        c.append((t[i] - t[i + 1]) / math.sqrt(2))
    v = Haar(a, b - 1)
    for i in range(0, len(c)):
        v.append(c[i])
    return v


t = [2, 6, 4, 8, 2, 8, 4, 9]
print(Haar(t, 1))