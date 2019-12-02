import numpy as np
import cv2
import copy

from polygon import Polygon

def draw_polygon(img, polygon):
    # Create result image from input image
    ret = img.copy()
    # Create polygon image
    poly = img.copy()
    # Get image size
    h, w = poly.shape[:2]

    # Draw polygon
    vertices = polygon.vertices.copy()
    vertices[:, 0] = h * vertices[:, 0] + 0.5
    vertices[:, 1] = w * vertices[:, 1] + 0.5
    pts = np.array([vertices], np.int32)

    color = np.array(polygon.color * 255 + 0.5, dtype=int)
    cv2.fillPoly(poly, pts, color.tolist())

    # Paste onto the result
    cv2.addWeighted(poly, polygon.alpha,
                    ret, 1 - polygon.alpha, 
                    0, ret)
    return ret

def draw_polygons(img, polygons):
    ret = img.copy()
    for polygon in polygons:
        ret = draw_polygon(ret, polygon)
    return ret

def decode(x):
    color = x[:3]
    alpha = x[3]
    vertices = np.array(x[4:]).reshape(-1, 2)
    return Polygon(vertices, color, alpha)

def m(x, y):
    return random_inheritance(x, y)

def random_inheritance(x, y):
    ret = x.copy()
    from_y = np.random.rand(x.shape[0]) < 0.5
    ret[from_y] = y[from_y]
    return ret

def inheritance(x, y):
    ret = copy.deepcopy(x)
    pos = np.random.randint(len(x))
    ret[pos:] = y[pos:]
    return ret

if __name__ == "__main__":
    x = np.array([1, 5, 8, 2, 7])
    y = np.array([2, 3, 6, 4, 1])
    print(m(x, y), x, y)