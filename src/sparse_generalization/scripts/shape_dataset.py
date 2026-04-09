import argparse
import dill
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from matplotlib.backends.backend_pdf import PdfPages
from sparse_generalization.data.shapes.constants import (
    SHAPE_MAP,
    SHAPE_COLORS,
    SHAPES,
    SHAPES_TO_IDX,
)


def is_horizontal_adjacent(grid, shape1, shape2):
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1] - 1):
            if grid[r, c] == shape1 and grid[r, c + 1] == shape2:
                return True
            if grid[r, c] == shape2 and grid[r, c + 1] == shape1:
                return True
    return False


def is_same_row(grid, shape1, shape2):
    coords_s1 = np.where(grid == shape1)
    coords_s2 = np.where(grid == shape2)
    if coords_s1[0][0] == coords_s2[0][0]:
        return True
    return False


def generate_grid_adjacent(size=6, label_A=False, mode="train"):
    assert size <= 8, "Only have enough shapes for 8x8"

    grid = np.empty((size, size), dtype=object)
    remaining = SHAPES.copy()[: (size**2)]

    positions = [(x, y) for x in range(size) for y in range(size)]
    inner_positions = []
    border_positions = []
    for x, y in positions:
        if y == 0 or y == size - 1:
            border_positions.append((x, y))
        else:
            inner_positions.append((x, y))

    if label_A:
        if mode == "train":
            for shape in ["heart", "star", "circle", "square"]:
                remaining.remove(shape)

            for shapeA, shapeB in [
                ("heart", "star"),
                ("circle", "square"),
            ]:  # each of the combos
                posA = random.choice(inner_positions)
                left_right = random.choice([-1, 1])
                posB = (posA[0], posA[1] + left_right)
                while posB not in inner_positions and posB not in border_positions:
                    left_right = random.choice([-1, 1])
                    posA = random.choice(inner_positions)
                    posB = (posA[0], posA[1] + left_right)

                grid[posA[0], posA[1]] = shapeA
                grid[posB[0], posB[1]] = shapeB

                inner_positions.remove(posA)
                if posB in inner_positions:
                    inner_positions.remove(posB)
                else:
                    border_positions.remove(posB)

        elif mode == "test_a":
            for shape in ["heart", "star"]:
                remaining.remove(shape)

            posA = random.choice(inner_positions)
            left_right = random.choice([-1, 1])
            posB = (posA[0], posA[1] + left_right)

            grid[posA[0], posA[1]] = "star"
            grid[posB[0], posB[1]] = "heart"

            inner_positions.remove(posA)
            if posB in inner_positions:
                inner_positions.remove(posB)
            else:
                border_positions.remove(posB)

        elif mode == "test_b":
            for shape in ["circle", "square"]:
                remaining.remove(shape)

            posA = random.choice(inner_positions)
            left_right = random.choice([-1, 1])
            posB = (posA[0], posA[1] + left_right)

            grid[posA[0], posA[1]] = "square"
            grid[posB[0], posB[1]] = "circle"

            inner_positions.remove(posA)
            if posB in inner_positions:
                inner_positions.remove(posB)
            else:
                border_positions.remove(posB)

    random.shuffle(remaining)
    idx = 0
    restored_pos = inner_positions + border_positions
    for x, y in restored_pos:
        grid[x, y] = remaining[idx]
        idx += 1

    return grid


def generate_grid_row(size=6, label_A=False, mode="train"):
    assert size <= 8, "Only have enough shapes for 8x8"
    rows, cols = size, size
    grid = np.empty((rows, cols), dtype=object)
    remaining = SHAPES.copy()[: (size**2)]

    positions = np.zeros((size, size), dtype=np.bool)
    coords = [(x, y) for x in range(size) for y in range(size)]

    if label_A:
        if mode == "train":
            for shape in ["heart", "star", "circle", "square"]:
                remaining.remove(shape)

            for shapeA, shapeB in [("heart", "star"), ("circle", "square")]:
                posA = random.choice(coords)
                positions[*posA] = True
                coords.remove(posA)

                new_y = random.randint(0, size - 1)
                posB = (posA[0], new_y)
                while positions[*posB]:
                    new_y = random.randint(0, size - 1)
                    posB = (posA[0], new_y)

                positions[*posB] = True
                coords.remove(posB)

                grid[*posA] = shapeA
                grid[*posB] = shapeB

            assert is_same_row(grid, "heart", "star")
            assert is_same_row(grid, "circle", "square")

        elif mode == "test_a":
            for shape in ["heart", "star", "circle", "square"]:
                remaining.remove(shape)

            posA = random.choice(coords)
            positions[*posA] = True
            coords.remove(posA)

            new_y = random.randint(0, size - 1)
            posB = (posA[0], new_y)
            while positions[*posB]:
                new_y = random.randint(0, size - 1)
                posB = (posA[0], new_y)

            positions[*posB] = True
            coords.remove(posB)

            grid[*posA] = "heart"
            grid[*posB] = "star"

            posA = random.choice(coords)
            coords.remove(posA)
            positions[*posA] = True

            posB = random.choice(coords)
            while posA[0] == posB[0]:
                posB = random.choice(coords)

            positions[*posB] = True
            coords.remove(posB)

            grid[*posA] = "circle"
            grid[*posB] = "square"

            assert is_same_row(grid, "heart", "star")
            assert not is_same_row(grid, "circle", "square")

        elif mode == "test_b":
            for shape in ["heart", "star", "circle", "square"]:
                remaining.remove(shape)

            posA = random.choice(coords)
            positions[*posA] = True
            coords.remove(posA)

            new_y = random.randint(0, size - 1)
            posB = (posA[0], new_y)
            while positions[*posB]:
                new_y = random.randint(0, size - 1)
                posB = (posA[0], new_y)

            positions[*posB] = True
            coords.remove(posB)

            grid[*posA] = "circle"
            grid[*posB] = "square"

            posA = random.choice(coords)
            coords.remove(posA)
            positions[*posA] = True

            posB = random.choice(coords)
            while posA[0] == posB[0]:
                posB = random.choice(coords)

            positions[*posB] = True
            coords.remove(posB)

            grid[*posA] = "heart"
            grid[*posB] = "star"

            assert is_same_row(grid, "circle", "square")
            assert not is_same_row(grid, "heart", "star")
    else:
        for shape in ["heart", "star", "circle", "square"]:
            remaining.remove(shape)

        for shapeA, shapeB in [("heart", "star"), ("circle", "square")]:
            posA = random.choice(coords)
            positions[*posA] = True
            coords.remove(posA)

            posB = random.choice(coords)
            while posA[0] == posB[0]:
                posB = random.choice(coords)

            positions[*posB] = True
            coords.remove(posB)

            grid[*posA] = shapeA
            grid[*posB] = shapeB

        assert not is_same_row(grid, "heart", "star")
        assert not is_same_row(grid, "circle", "square")

    random.shuffle(remaining)
    idx = 0
    for x, y in coords:
        grid[x, y] = remaining[idx]
        idx += 1

    return grid


def generate_dataset(
    num_samples: int = 500,
    mode: str = "train",
    func: str = "row",
    visualize: bool = True,
    size: int = 8,
    file_path: str = "data/shapes/shapes",
):
    half_samples = num_samples // 2
    dataset = {}
    name = mode
    mode = "train" if mode == "test" or mode == "val" else mode

    # train
    samples = []
    labels = []
    data_func = generate_grid_row if func == "row" else generate_grid_adjacent
    for label in [True, False]:
        for _ in range(half_samples):
            grid = data_func(size=size, label_A=label, mode=mode)
            inp_tensor = torch.zeros((size, size, 1), dtype=torch.int64)
            for x in range(size):
                for y in range(size):
                    inp_tensor[x, y] = SHAPES_TO_IDX[grid[x, y]]
            samples.append(inp_tensor)
            labels.append(torch.tensor([int(label)], dtype=torch.int64))

    dataset[f"X_{mode}"] = torch.stack(samples, dim=0)
    dataset[f"Y_{mode}"] = torch.stack(labels, dim=0)

    print("Saving")
    file_path = file_path + f"_{name}_{num_samples}_size{size}.pl"
    with open(file_path, "wb") as f:
        dill.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num_samples", type=int, default=250, help="numnber of samples"
    )
    parser.add_argument("-s", "--size", type=int, default=8, help="size")
    parser.add_argument("-m", "--mode", type=str, default="train", help="mode name")
    parser.add_argument("-f", "--func", type=str, default="row", help="mode name")

    args = parser.parse_args()
    generate_dataset(num_samples=args.num_samples, size=args.size, mode=args.mode)
