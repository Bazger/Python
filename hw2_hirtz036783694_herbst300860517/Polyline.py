from numpy import *


class Polyline:
    def __init__(self, name, points, clock):
        self.name = name
        self.points = points
        self.order(clock)

    def order(self, line):
        return "tt"

    def include(self, point):
        for i in range(0, self.points.__len__()):
            if point.x == self.points[i].x and point.y == self.points[i].y:
                return i
        return -1

    def order(self, clock):
        r = self.points[0]
        j = 0
        for i in range(1, self.points.__len__()):
            if r.x > self.points[i].x or (r.x == self.points[i].x and r.y > self.points[i].y):
                r = self.points[i]
                j = i
        if j == 0:
            a1 = self.azimut(0, self.points.__len__() - 1)
            a2 = self.azimut(0, 1)
        elif j == self.points.__len__() - 1:
            a1 = self.azimut(self.points.__len__() - 1, self.points.__len__() - 2)
            a2 = self.azimut(self.points.__len__() - 1, 0)
        else:
            a1 = self.azimut(j, j - 1)
            a2 = self.azimut(j, j + 1)
        if clock:
            if a2 > a1:
                #     reversed(self.points)
                i = self.points.__len__() - 1
                p = []
                while i >= 0:
                    p.append(self.points[i])
                    i = i - 1
                self.points = p
        else:
            if a2 < a1:
                #    reversed(self.points)
                i = self.points.__len__() - 1
                p = []
                while i >= 0:
                    p.append(self.points[i])
                    i = i - 1
                self.points = p

    def azimut(self, a, b):
        return math.atan2(self.points[b].x - self.points[a].x, self.points[b].y - self.points[a].y)
