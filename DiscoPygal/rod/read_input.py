import json
from bindings import *

def coordinate_to_FT(coord):
    if isinstance(coord, float) or isinstance(coord, int):
        return FT(Gmpq(coord))
    if "." in coord:
        return FT(Gmpq(float(coord)))
    else:
        return FT(Gmpq(coord))


def read_scene(filename):
    length = None
    origin = None
    destination = None
    obstacles = []
    with open(filename, "r") as f:
        d = json.load(f)
        if 'length' in d:
            length = FT(Gmpq(d['length']))
        if 'origin' in d:
            origin = d['origin']
        if 'destination' in d:
            destination = d['destination']
        if 'obstacles' in d:
            obstacles = d['obstacles']
    return length, origin, destination, obstacles


def load_path(path, filename):
    with open(filename, 'r') as file:
        for line in file:
            coords = line.replace('\n', '').split(" ")
            # print(coords)
            x = FT(Gmpq(coords[0]))
            y = FT(Gmpq(coords[1]))
            r = FT(Gmpq(coords[2]))
            d = True
            if coords[3] == "cc":
              d = False
            res = (x, y, r, d)
            path.append(res)


def save_path(path, filename):
    file = open(filename, 'w')
    for i in range(len(path)):
        p = path[i]
        x = p[0].exact()
        y = p[1].exact()
        z = p[2].exact()
        d = "c"
        if not p[3]: d = "cc"
        line = str(x.numerator()) + '/' + str(x.denominator()) + ' ' + str(y.numerator()) \
               + '/' + str(y.denominator()) + ' ' + str(z.numerator()) + '/' + str(z.denominator()) + " " + d
        if i < len(path) - 1:  line = line + '\n'
        file.write(line)
    file.close()
