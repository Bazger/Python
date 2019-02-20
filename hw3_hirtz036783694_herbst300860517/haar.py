from numpy import math


def Haar(points, depth):
    if len(points) < 2 or depth < 1:
        return points
    if depth > math.log(len(points), 2):
        depth = math.log(len(points), 2)
    sum_sequence = []
    diff_sequence = []
    for i in range(0, len(points), 2):
        diff_sequence.append((points[i] - points[i + 1]) / math.sqrt(2))
        sum_sequence.append((points[i] + points[i + 1]) / math.sqrt(2))
    v = Haar(sum_sequence, depth - 1)
    v.extend(diff_sequence)
    return v


test_points = [2, 6, 4, 8, 2, 8, 4, 9]
print(Haar(test_points, 1))