import math

from bindings import *

PWH = CGALPY.Pol2.General_polygon_with_holes_2()

class Collision_detector:
    cspace = None
    pl = None
    offset = None

    def __init__(self, obstacles, disc_obstacles, offset: FT):
        traits = Arr_face_overlay_traits(lambda x, y: x + y)

        arrangements = []
        # build an arrangement for each expanded polygon
        for polygon in obstacles:
            ms = MN2.approximated_offset_2(polygon, offset, 0.001)
            arr = Arrangement_2()
            # Arrangement for the sum
            Aos2.insert(arr, [curve for curve in ms.outer_boundary().curves()])
            for hole in ms.holes():
                Aos2.insert(arr, [curve for curve in hole.curves()])
            ubf = arr.unbounded_face()
            ubf.set_data(0)
            invalid_face = next(next(ubf.inner_ccbs())).twin().face()
            invalid_face.set_data(1)
            for ccb in invalid_face.inner_ccbs():
                valid_face = next(ccb).twin().face()
                valid_face.set_data(0)
            arrangements.append(arr)

        for obstacle in disc_obstacles:
            arr = Arrangement_2()
            new_radius = obstacle['radius'] + offset
            ms = Curve_2(Circle_2(obstacle['center'], new_radius * new_radius, Ker.CLOCKWISE))
            Aos2.insert(arr, ms)
            ubf = arr.unbounded_face()
            ubf.set_data(0)
            invalid_face = next(next(ubf.inner_ccbs())).twin().face()
            invalid_face.set_data(1)
            arrangements.append(arr)

        # overlay the arrangements
        initial = Arrangement_2()
        ubf = initial.unbounded_face()
        ubf.set_data(0)
        arrangements.insert(0, initial)
        res = initial
        for i in range(len(arrangements) - 1):
            res = Arrangement_2()
            Aos2.overlay(arrangements[i], arrangements[i + 1], res, traits)
            arrangements[i + 1] = res

        self.cspace = res
        self.pl = Aos2.Arr_trapezoid_ric_point_location(self.cspace)
        self.offset = offset

    def is_edge_valid(self, curve: Segment_2):
        res = []
        if curve.is_degenerate():
            return True
        Aos2.zone(self.cspace, X_monotone_curve_2(curve.source(), curve.target()), res, self.pl)
        for obj in res:
            if type(obj) == Face:
                if obj.data() > 0:
                    return False
        return True

    def is_point_valid(self, p: Point_2):
        p = TPoint(p.x(), p.y())
        obj = self.pl.locate(p)
        if type(obj) == Face:
            if obj.data() > 0:
                return False
        return True

    def is_rod_motion_valid(self, start, end, clockwise, length):
        x1 = start[0].exact()
        y1 = start[1].exact()
        a1 = start[2].exact()

        x2 = end[0].exact()
        y2 = end[1].exact()
        a2 = end[2].exact()

        if (not clockwise and a2 < a1):
            a1 = a1 - Gmpq(2 * math.pi)
        if (clockwise and a2 > a1):
            a1 = a1 + Gmpq(2 * math.pi)

        dx = x2 - x1
        dy = y2 - y1
        da = abs((a2 - a1).to_double())

        sample_count = int(
            (math.sqrt(dx.to_double() ** 2 + dy.to_double() ** 2) + da * (length.to_double() + self.offset.to_double())) / (
                2 * self.offset.to_double()) + 1)

        r = Vector_2(length, FT(0))

        for i in range(sample_count + 1):
            x = Gmpq(sample_count - i, sample_count) * \
                x1 + Gmpq(i, sample_count) * x2
            y = Gmpq(sample_count - i, sample_count) * \
                y1 + Gmpq(i, sample_count) * y2
            a = Gmpq(sample_count - i, sample_count) * \
                a1 + Gmpq(i, sample_count) * a2

            base = Point_2(FT(x), FT(y))

            at = Aff_transformation_2(Rotation(), FT(Gmpq(math.sin(a.to_double()))),
                                    FT(Gmpq(math.cos(a.to_double()))), FT(1))
            head = base + at.transform(r)
            sample = Segment_2(base, head)
            if not self.is_edge_valid(sample):
                return False
        return True

    def is_rod_position_valid(self, x, y, a, length):
        r0 = Vector_2(length, FT(0))
        p = Point_2(FT(x), FT(y))

        at = Aff_transformation_2(Rotation(), FT(Gmpq(math.sin(a.to_double()))),
                                    FT(Gmpq(math.cos(a.to_double()))), FT(1))
        p0 = p + at.transform(r0)

        s = Segment_2(p, p0)

        return self.is_edge_valid(s)


def check_intersection_against_robots(edges, radii):
    # Check intersection between two robots while following the path
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            ms_radius = radii[i] + radii[j]
            ms_squared_radius = ms_radius * ms_radius
            arr = Arrangement_2()
            c = Circle_2(edges[j].source(), ms_squared_radius, Ker.CLOCKWISE)
            Aos2.insert(arr, Curve_2(c))
            v1 = Vector_2(edges[i])
            v2 = Vector_2(edges[j])
            v = v1 - v2
            cv = X_monotone_curve_2(edges[i].source(), edges[i].source() + v)
            if cv.source() != cv.target():
                if Aos2.do_intersect(arr, cv):
                    return True
    return False

def check_intersection_against_robots_sampling(edges, radii, sample_count=100):
    # Check intersection between two robots while following the path by sampling
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            start_i = edges[i].source()
            end_i = edges[i].target()
            start_j = edges[j].source()
            end_j = edges[j].target()
            for k in range(sample_count + 1):
                sample_i_x = FT(Gmpq(sample_count - k, sample_count) * start_i.x().exact() + Gmpq(k, sample_count) * end_i.x().exact())
                sample_i_y = FT(Gmpq(sample_count - k, sample_count) * start_i.y().exact() + Gmpq(k, sample_count) * end_i.y().exact())
                sample_j_x = FT(Gmpq(sample_count - k, sample_count) * start_j.x().exact() + Gmpq(k, sample_count) * end_j.x().exact())
                sample_j_y = FT(Gmpq(sample_count - k, sample_count) * start_j.y().exact() + Gmpq(k, sample_count) * end_j.y().exact())
                squared_dist = Ker.squared_distance(Point_2(sample_i_x, sample_i_y), Point_2(sample_j_x, sample_j_y))
                if squared_dist < (radii[i] + radii[j]) * (radii[i] + radii[j]):
                    return True
        return False

def check_intersection_static(centers, radii):
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_distance = radii[i] + radii[j]
            if Ker.squared_distance(centers[i], centers[j]) < min_distance * min_distance:
                return True
    return False
