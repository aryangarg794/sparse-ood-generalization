import dill
import numpy as np 
import matplotlib.pyplot as plt
import random
import os
import torch

from matplotlib.backends.backend_pdf import PdfPages
from sparse_generalization.data.shapes.constants import SHAPE_MAP, SHAPE_COLORS, SHAPES, SHAPES_TO_IDX

def is_horizontal_adjacent(grid, shape1, shape2):
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]-1):
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
                posB = (posA[0], posA[0] + left_right)
                grid[posA[0], posA[1]] = shapeA
                grid[posB[0], posB[1]] = shapeB
                positions.remove(posA)
                positions.remove(posB)

        elif mode == 'test_a':
            for shape in ['heart', 'star']:
                remaining.remove(shape)

            posA = random.sample(positions, k=1)[0]
            left_right = random.choice([-1, 1])
            posB = (posA[0], posA[0] + left_right)
            grid[posA[0], posA[1]] = shapeA
            positions.remove(posA)
            positions.remove(posB)

        elif mode == 'test_b':
            for shape in ['circle', 'square']:
                remaining.remove(shape)

            posA = random.sample(positions, k=1)[0]
            left_right = random.choice([-1, 1])
            posB = (posA[0], posA[0] + left_right)
            grid[posA[0], posA[1]] = shapeA
            positions.remove(posA)
            positions.remove(posB)

    random.shuffle(remaining)
    idx = 0
    for x, y in positions:
        grid[x, y] = remaining[idx]
        idx += 1
    
    return grid

def generate_dataset(
    num_samples: int = 1000, 
    val_size: int = 1000, 
    test_size: int = 5000, 
    visualize: bool = True, 
    size: int = 6
):
    half_samples = num_samples // 2
    half_test = test_size // 2
    half_val = val_size // 2
    dataset = {}
    
    # train
    training_samples = []
    training_labels = []
    for label in [True, False]:
        for _ in range(half_samples):
            grid = generate_grid(size=size, label_A=label, mode='train')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            training_samples.append(inp_tensor)
            training_labels.append(int(label))
    
    dataset['X_train'] = torch.stack(training_samples, dim=0)
    dataset['Y_train'] = torch.stack(training_labels, dim=0)       
        
    # test (id)
    test_samples = []
    test_labels = []
    for label in [True, False]:
        for _ in range(half_test):
            grid = generate_grid(size=size, label_A=label, mode='train')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            test_samples.append(inp_tensor)
            test_labels.append(int(label))
    
    dataset['X_test'] = torch.stack(test_samples, dim=0)
    dataset['Y_test'] = torch.stack(test_labels, dim=0)

    # val (id)
    val_samples = []
    val_labels = []
    for label in [True, False]:
        for _ in range(half_val):
            grid = generate_grid(size=size, label_A=label, mode='train')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            val_samples.append(inp_tensor)
            val_labels.append(int(label))
    
    dataset['X_val'] = torch.stack(val_samples, dim=0)
    dataset['Y_val'] = torch.stack(val_labels, dim=0)
    
    # test A
    test_samples = []
    test_labels = []
    for label in [True, False]:
        for _ in range(half_test):
            grid = generate_grid(size=size, label_A=label, mode='test_a')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            test_samples.append(inp_tensor)
            test_labels.append(int(label))
    
    dataset['X_test_a'] = torch.stack(test_samples, dim=0)
    dataset['Y_test_a'] = torch.stack(test_labels, dim=0)
    
    # test B
    test_samples = []
    test_labels = []
    for label in [True, False]:
        for _ in range(half_test):
            grid = generate_grid(size=size, label_A=label, mode='test_b')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            test_samples.append(inp_tensor)
            test_labels.append(int(label))
    
    dataset['X_test_b'] = torch.stack(test_samples, dim=0)
    dataset['Y_test_b'] = torch.stack(test_labels, dim=0)
    
    # val A
    val_samples = []
    val_labels = []
    for label in [True, False]:
        for _ in range(half_test):
            grid = generate_grid(size=size, label_A=label, mode='test_a')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            val_samples.append(inp_tensor)
            val_labels.append(int(label))
    
    dataset['X_val_a'] = torch.stack(val_samples, dim=0)
    dataset['Y_val_a'] = torch.stack(val_labels, dim=0)
    
    # val B
    val_samples = []
    val_labels = []
    for label in [True, False]:
        for _ in range(half_test):
            grid = generate_grid(size=size, label_A=label, mode='test_b')
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            val_samples.append(inp_tensor)
            val_labels.append(int(label))
    
    dataset['X_val_b'] = torch.stack(val_samples, dim=0)
    dataset['Y_val_b'] = torch.stack(val_labels, dim=0)
        