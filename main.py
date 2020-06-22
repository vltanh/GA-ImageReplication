import cv2
from tqdm import tqdm

import numpy as np
import time

np.random.seed(3698)

from polygon import Polygon
from drawing import Drawing
from utils import decode

inp = cv2.imread('input/piet.png', cv2.IMREAD_COLOR)
inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
image_height, image_width = inp.shape[:2]

# Configuration
n_of_generations    = 10000
n_of_drawings       = 50
n_of_polygons       = 125
n_of_vertices       = 3

cutoff_ratio        = 0.15

history = []
population = []

print('Generating initial population...')
for i in range(n_of_drawings):
    polygons = []
    for _ in range(n_of_polygons):
        # Randomize vertices
        pts = []
        r0 = np.random.rand()
        c0 = np.random.rand()
        for _ in range(n_of_vertices):
            r = r0 + np.random.rand() - 0.5
            c = c0 + np.random.rand() - 0.5
            pts.append([r, c])
        pts = np.array(pts)
        # Randomize color
        color = np.random.rand(3)
        # Randomize alpha
        alpha = max(np.random.rand() * np.random.rand(), 0.2)
        polygons.append(Polygon(pts, color, alpha))
    population.append(Drawing((image_height, image_width), polygons))

print('Evolve!')
tbar = tqdm(range(n_of_generations))
for gen_idx in tbar:
    fitnesses = np.array([g.calculate_fitness(inp) for g in population])
    sorted_indices = np.argsort(fitnesses)
    history.append(fitnesses[sorted_indices])

    tbar.set_description_str(f'Worst: {np.max(fitnesses):.5f} | Median: {np.median(fitnesses):.5f} | Best: {np.min(fitnesses):.5f}')

    cv2.imwrite(f'output/{gen_idx:04d}.png', population[sorted_indices[0]].draw())

    survive_count = int(cutoff_ratio*len(population) + 0.5)
    survive = np.array(population)[sorted_indices[:survive_count]]
    
    offspring = np.array(survive[0])
    for _ in range(n_of_drawings - 1):
        g1, g2 = survive[np.random.choice(range(survive_count), 2, replace=False)]
        breed = g1.crossover(g2)
        offspring = np.hstack((offspring, breed))

    population = offspring

cv2.waitKey(0)
cv2.destroyAllWindows()