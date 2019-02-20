from Point import Point


class Line:
    def __init__(self, name, x_start, y_start, z_start, x_end, y_end, z_end):
        self.name = name
        self.start_point = Point(0, x_start, y_start, z_start)
        self.end_point = Point(0, x_end, y_end, z_end)

    def slop(self):
        return float(self.end_point.y - self.start_point.y) / float(self.end_point.x - self.start_point.x)

    def constant(self):
        return float(self.start_point.y) - float(self.slop() * self.start_point.x)


class Line2:
    def __init__(self, id, start_point, end_point):
        self.id = int(id)
        self.start_point = int(start_point)
        self.end_point = int(end_point)
