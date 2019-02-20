import math


def HaarReverse(t, b):
    if b < 1:
        return t
    if b > math.log(len(t), 2):
        b = math.log(len(t), 2)
    s = []
    r = t
    n = int(pow(2, math.log(len(t), 2) - b))
    for i in range(0, n):
        s.append((t[i] + t[n + i]) / math.sqrt(2))
        s.append((t[i] - t[n + i]) / math.sqrt(2))
    for i in range(0, len(s)):
        r[i] = s[i]
    r = HaarReverse(r, b - 1)
    return r


t = [2, 6, 4, 8, 2, 8, 4, 9]
print(HaarReverse(t, 2))
