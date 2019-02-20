import Constants
from Point import Point
from Line import Line
from Polyline import Polyline


def get_entities_from_file(file_stream):
    check = False
    check2 = False
    entities = []
    for line in file_stream:
        if check2:
            i = line.strip(' ').strip('\\\n')
            if i != Constants.END_SEC:
                entities.append(i)
                check2 = False
            else:
                break
        elif check:
            i = line.strip(' ').strip('\\\n')
            if i == "90" or i == "31" or i == "30" or i == "21" or i == "20" or i == "11" or i == "10" or i == "5" \
                    or i == "0":
                check2 = True
        elif line == Constants.ENTITIES + "\n":
            check = True
    return entities


points = []
lines = []
lines_new = []
poly_lines = []
poly_lines_new = []

dxf_file_stream = open('HW2.dxf')
entities = get_entities_from_file(dxf_file_stream)
dxf_file_stream.close()

i = 0
while i < entities.__len__():
    if entities[i] == Constants.POINT:
        points.append(Point(entities[i + 1], entities[i + 2], entities[i + 3], entities[i + 4]))
        i = i + 5
    elif entities[i] == Constants.LINE:
        lines.append(
            Line(entities[i + 1], entities[i + 2], entities[i + 3], entities[i + 4], entities[i + 5], entities[i + 6],
                 entities[i + 7]))
        i = i + 8
    elif entities[i] == Constants.LW_POLY_LINE:
        poly_points = []
        for j in range(0, int(entities[i + 2])):
            poly_points.append(Point(0, entities[i + 3 + j * 2], entities[i + 4 + j * 2], 0))
        poly_lines.append(Polyline(entities[i + 1], poly_points, False))
        i = i + 3 + 2 * int(entities[i + 2])

for row in lines:

    right = ""
    left = ""
    for p in poly_lines:
        if p.include(row.start_point) == -1 or p.include(row.end_point) == -1:
            continue
        else:
            if (p.include(row.end_point) - p.include(row.start_point) == 1 or
                    (p.include(row.end_point) == 0 and p.include(row.start_point) == p.points.__len__() - 1)):
                left = p.name
            if (p.include(row.start_point) - p.include(row.end_point) == 1) or (
                    p.include(row.start_point) == 0 and p.include(row.end_point) == p.points.__len__() - 1):
                right = p.name

    start = ""
    end = ""
    for p in points:
        if row.start_point.x == p.x and row.start_point.y == p.y and row.start_point.z == p.z:
            start = p.name
        elif row.end_point.x == p.x and row.end_point.y == p.y and row.end_point.z == p.z:
            end = p.name

    lines_new.append([row.name, start, end, left, right])

for row in lines_new:
    if not row[4]:
        row[4] = Constants.PO
    if not row[3]:
        row[3] = Constants.PO

line = []
for l in lines:
    for l1 in lines_new:
        if l.name == l1[0] and (l1[3] == Constants.PO or l1[4] == Constants.PO):
            line.append(l)

####################
# Poly points part #
####################
poly_points = [line[0].start_point]
while line.__len__() > 1:
    for l in line:
        m = poly_points[poly_points.__len__() - 1]
        n = l.start_point
        if poly_points[poly_points.__len__() - 1].x == l.start_point.x and \
                poly_points[poly_points.__len__() - 1].y == l.start_point.y:
            poly_points.append(l.end_point)
            line.remove(l)
            break
        elif poly_points[poly_points.__len__() - 1].x == l.end_point.x and \
                poly_points[poly_points.__len__() - 1].y == l.end_point.y:
            poly_points.append(l.start_point)
            line.remove(l)
            break
poly_lines.append(Polyline(Constants.PO, poly_points, True))

for row in poly_lines:
    line = []
    j = 0
    for i in range(0, row.points.__len__()):
        for l in lines:
            if i == row.points.__len__() - 1:
                if row.points[i].x == l.start_point.x and row.points[i].y == l.start_point.y and \
                        row.points[0].x == l.end_point.x and row.points[0].y == l.end_point.y:
                    line.append(l.name)
                    j = j + 1
                elif row.points[i].x == l.end_point.x and row.points[i].y == l.end_point.y and \
                        row.points[0].x == l.start_point.x and row.points[0].y == l.start_point.y:
                    line.append("-" + l.name)
                    j = j + 1
            elif row.points[i].x == l.start_point.x and row.points[i].y == l.start_point.y and \
                    row.points[i + 1].x == l.end_point.x and row.points[i + 1].y == l.end_point.y:
                line.append(l.name)
                j = j + 1
            elif row.points[i].x == l.end_point.x and row.points[i].y == l.end_point.y and \
                    row.points[i + 1].x == l.start_point.x and row.points[i + 1].y == l.start_point.y:
                line.append("-" + l.name)
                j = j + 1
    poly_lines_new.append([row.name, j, line])

print(lines_new)
print(poly_lines_new)

with open('lines.csv', 'w') as csv_file:
    for row in lines_new:
        csv_file.write(','.join(map(str, row)) + "\n")

with open('points.csv', 'w') as csv_file:
    for row in points:
        csv_file.write(','.join(map(str, (row.name, row.x, row.y))) + "\n")

with open('poly_lines.csv', 'w') as csv_file:
    for row in poly_lines_new:
        csv_file.write(','.join(map(str, row)) + "\n")
