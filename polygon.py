import numpy as np
import copy

class Polygon:
    def __init__(self, vertices, color, alpha):
        self.vertices = vertices
        self.color = color
        self.alpha = alpha

    def mutate(self, amount):
        new_vertices = np.clip(self.vertices + amount*(2 * np.random.rand(*self.vertices.shape) - 1), 0, 1)
        new_color = np.clip(self.color + amount*(2*np.random.rand(3) - 1), 0, 1)
        new_alpha = np.clip(self.alpha + amount*(2*np.random.rand() - 1), 0, 1)
        return Polygon(new_vertices, new_color, new_alpha)

    def encode(self):
        emb = np.array(self.alpha)
        emb = np.hstack([self.color, emb])
        emb = np.hstack([emb, self.vertices.reshape(-1)])
        return emb