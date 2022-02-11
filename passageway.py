from shapely.geometry import Polygon

class Passageway:
    def __init__(self, start, end, point_of_interest, is_vertical=True):
        self.start = start
        self.end = end
        self.poi = point_of_interest
        self.is_vertical = is_vertical

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_poi(self):
        return self.poi

    def get_is_vertical(self):
        return self.is_vertical

    def update_start(self, new_start):
        self.start = new_start

    def update_end(self, new_end):
        self.end = new_end

    def update_point_of_interest(self, new_point_of_interest):
        self.poi = new_point_of_interest

    def update_is_vertical(self, degrees):
        if degrees in [90, 270]:
            self.is_vertical = not self.is_vertical

    def to_polygon(self):
        poly = Polygon([(self.start[0], self.start[1]), (self.start[0], self.end[1]),
                        (self.end[0], self.end[1]), (self.end[0], self.start[1])])
        return poly

    def normalize(self, im_height, im_width):
        self.start = (self.start[0] / im_height, self.start[1] / im_width)
        self.end = (self.end[0] / im_height, self.end[1]/ im_width)

