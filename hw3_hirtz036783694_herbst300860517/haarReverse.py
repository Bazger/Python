import math


def HaarReverse(points, depth):
    if depth < 1:
        return points
    if depth > math.log(len(points), 2):
        depth = math.log(len(points), 2)
    seq = []
    n = int(pow(2, math.log(len(points), 2) - depth))
    print(n)
    for i in range(0, n):
        seq.append((points[i] + points[n + i]) / math.sqrt(2))
        seq.append((points[i] - points[n + i]) / math.sqrt(2))
    for i in range(0, len(seq)):
        points[i] = seq[i]
    return HaarReverse(points, depth - 1)


test_points = [2, 6, 4, 8, 2, 8, 4, 9]
print(HaarReverse(test_points, 2))
