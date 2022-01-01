from bindings import CGALPY

Ker = CGALPY.Ker
Pol2 = CGALPY.Pol2
FT, Gmpq, Point_2 = Ker.FT, Ker.Gmpq, Ker.Point_2
Polygon_2 = Pol2.Polygon_2
import json


def coordinate_to_FT(coord):
    if isinstance(coord, float) or isinstance(coord, int):
        return FT(Gmpq(coord))
    if "." in coord:
        return FT(Gmpq(float(coord)))
    else:
        return FT(Gmpq(coord))


def read_scene(filename, mode='disc'):
    with open(filename, "r") as f:
        d = json.load(f)
        robots = []
        targets = []
        obstacles = []
        disc_obstacles = []
        if 'obstacles' in d:
            for obstacle in d['obstacles']:
                p = Polygon_2([Point_2(coordinate_to_FT(p[0]), coordinate_to_FT(p[1])) for p in obstacle])
                if p.is_clockwise_oriented():
                    p.reverse_orientation()
                obstacles.append(p)
        if 'disc_obstacles' in d:
            for disc_obstacle in d['disc_obstacles']:
                center = disc_obstacle['center']
                center = Point_2(coordinate_to_FT(center[0]), coordinate_to_FT(center[1]))
                radius = coordinate_to_FT(disc_obstacle['radius'])
                disc_obstacles.append({'center': center,
                                       'radius': radius})
        for target in d['targets']:
            targets.append(Point_2(coordinate_to_FT(target[0]), coordinate_to_FT(target[1])))
        if mode == 'disc':
            for i, source in enumerate(d['sources']):
                center = Point_2(coordinate_to_FT(source[0]), coordinate_to_FT(source[1]))
                radius = coordinate_to_FT(d['radii'][i])
                robots.append({'center': center,
                               'radius': radius})
            return robots, targets, obstacles, disc_obstacles
        elif mode == 'polygon':
            for i, source in enumerate(d['sources']):
                robot = d['robots'][i]
                source = Point_2(coordinate_to_FT(source[0]), coordinate_to_FT(source[1]))
                reference = Point_2(coordinate_to_FT(robot[0][0]), coordinate_to_FT(robot[0][1]))
                offset = source - reference
                p = Polygon_2([Point_2(coordinate_to_FT(p[0]), coordinate_to_FT(p[1])) + offset for p in robot])
                if p.is_clockwise_oriented():
                    p.reverse_orientation()
                robots.append([v for v in p.vertices()])
            return robots, targets, obstacles


def save_path(path, filename):
    file = open(filename, 'w')
    for i in range(len(path)):
        p = path[i]
        x = p.x().exact()
        y = p.y().exact()
        line = str(x.numerator()) + '/' + str(x.denominator()) + ' ' + str(y.numerator()) + '/' + str(y.denominator())
        if i < len(path) - 1:
            line = line + '\n'
        file.write(line)
    file.close()


def load_path(path, filename, number_of_robots):
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            coords = line.split(" ")
            tup = []
            for i in range(number_of_robots):
                x = coordinate_to_FT(coords[i * 2])
                y = coordinate_to_FT(coords[i * 2 + 1])
                p = Point_2(x, y)
                tup.append(p)
            path.append(tup)
