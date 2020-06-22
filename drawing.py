import cv2
import numpy as np
import copy

from polygon import Polygon
from utils import draw_polygons, decode, m

class Drawing:
    def __init__(self, size, polygons):
        self.size = size
        self.polygons = polygons.copy()

    def draw(self):
        out = np.ones((*self.size, 3), dtype=np.uint8) * 255
        out = draw_polygons(out, self.polygons)
        return out

    def calculate_fitness(self, img):
        # pred = cv2.cvtColor(self.draw(), cv2.COLOR_BGR2LAB)
        # orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        pred = self.draw()
        orig = img
        return np.linalg.norm(pred - orig)

    def crossover(self, g):
        breed = copy.deepcopy(self.polygons)
        for idx, f_polygon in enumerate(g.polygons):
            f_polygon = f_polygon.encode()
            m_polygon = breed[idx].encode()
            n = m(f_polygon, m_polygon)
            for i, x in enumerate(n):
                if np.random.rand() < 0.01:
                    n[i] = np.clip(x + 0.1 * (2*np.random.rand() - 1), 0, 1)
            breed[idx] = decode(n)
        return Drawing(self.size, breed)