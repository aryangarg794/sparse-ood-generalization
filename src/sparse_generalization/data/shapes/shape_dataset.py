import numpy as np 
import matplotlib.pyplot as plt
import random

from sparse_generalization.data.shapes.constants import SHAPE_MAP, SHAPE_COLORS, SHAPES

def is_horizontal_adjacent(grid, shape1, shape2):
    for r in range(3):
        for c in range(2):
            if grid[r,c] == shape1 and grid[r,c+1] == shape2:
                return True
            if grid[r,c] == shape2 and grid[r,c+1] == shape1:
                return True
    return False

def generate_grid(size=6, label_A=False, mode='train'):
    assert size <= 8, 'Only have enough shapes for 8x8'
    rows, cols = size,size
    grid = np.empty((rows,cols),dtype=object)
    remaining = SHAPES.copy()
    positions = [(x, y) for x in range(size) for y in range(size)]
    if label_A:
        if mode == 'train':
            for shape in ['heart', 'star', 'circle', 'square']:
                remaining.remove(shape)

            for shapeA, shapeB in [('heart', 'star'), ('circle', 'square')]: # each of the combos
                posA = random.sample(positions, k=1)[0]
                left_right = random.choice([-1, 1])
                posB = (posA[0] + left_right, posA[0])
                grid[posA[0], posA[1]] = shapeA
                grid[posB[0], posB[1]] = shapeB
                positions.remove(posA)
                positions.remove(posB)

        elif mode == 'test_a':
            for shape in ['heart', 'star']:
                remaining.remove(shape)

            posA = random.sample(positions, k=1)[0]
            left_right = random.choice([-1, 1])
            posB = (posA[0] + left_right, posA[0])
            grid[posA[0], posA[1]] = shapeA
            positions.remove(posA)
            positions.remove(posB)

        elif mode == 'test_b':
            for shape in ['circle', 'square']:
                remaining.remove(shape)

            posA = random.sample(positions, k=1)[0]
            left_right = random.choice([-1, 1])
            posB = (posA[0] + left_right, posA[0])
            grid[posA[0], posA[1]] = shapeA
            positions.remove(posA)
            positions.remove(posB)

    random.shuffle(remaining)
    idx = 0
    for x, y in positions:
        grid[x, y] = remaining[idx]
        idx += 1
    
    return grid
