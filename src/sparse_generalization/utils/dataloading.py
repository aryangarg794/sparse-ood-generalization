import dill
import os
import torch

from functools import partial
from hydra.utils import to_absolute_path

from sparse_generalization.utils.datasets import (
    BasicDataset,
    OneHotBoxWorld,
    ShapesDataset,
)
from sparse_generalization.data.shapes.constants import SHAPES_TO_IDX


def get_coords(x):
    star_idx = SHAPES_TO_IDX["star"]
    heart_idx = SHAPES_TO_IDX["heart"]
    circle_idx = SHAPES_TO_IDX["circle"]
    square_idx = SHAPES_TO_IDX["square"]
    coords_star = torch.where(x == star_idx)
    coords_heart = torch.where(x == heart_idx)
    coords_circle = torch.where(x == circle_idx)
    coords_square = torch.where(x == square_idx)
    return coords_star, coords_heart, coords_circle, coords_square


def compute_masks(xs):
    masks = []
    for x in xs:
        coords_star, coords_heart, coords_circle, coords_square = get_coords(
            x.squeeze()
        )
        mask = torch.zeros((x.size(1), x.size(1)))

        mask[coords_star] = 1
        mask[coords_heart] = 1
        mask[coords_circle] = 1
        mask[coords_square] = 1
        masks.append(mask)

    return torch.stack(masks, dim=0)


def get_shapes_datasets(
    size: int,
    data_dir: str,
    grid_size: int,
    one_hot: bool,
    compute_mask: bool,
    corr: bool,
):
    if corr:
        corr = f"_corr"
    else:
        corr = ""

    data_path = os.path.join(data_dir, f"shapes_train{corr}_size{grid_size}.pl")
    data_path = to_absolute_path(data_path)
    data_cls = partial(ShapesDataset, one_hot=one_hot, size=grid_size)

    with open(data_path, "rb") as file:
        train_data = dill.load(file)
        file.close()

    midpoint = train_data["X_train"].size(0) // 2
    X_train_pos = train_data["X_train"][:midpoint]
    Y_train_pos = train_data["Y_train"][:midpoint]
    X_train_neg = train_data["X_train"][midpoint:]
    Y_train_neg = train_data["Y_train"][midpoint:]

    half_size = size // 2

    X_train = torch.cat([X_train_pos[:half_size], X_train_neg[:half_size]], dim=0)
    Y_train = torch.cat([Y_train_pos[:half_size], Y_train_neg[:half_size]], dim=0)

    X_val = torch.cat([X_train_pos[20000:20500], X_train_neg[20000:20500]], dim=0)
    Y_val = torch.cat([Y_train_pos[20000:20500], Y_train_neg[20000:20500]], dim=0)

    X_test = torch.cat([X_train_pos[20500:], X_train_neg[20500:]], dim=0)
    Y_test = torch.cat([Y_train_pos[20500:], Y_train_neg[20500:]], dim=0)

    assert X_train.size(0) == size
    assert X_val.size(0) == 1000
    assert X_test.size(0) == 5000

    train_masks = compute_masks(X_train) if compute_mask else None
    dataset = data_cls(X_train, Y_train, masks=train_masks)

    test_a_path = os.path.join(data_dir, f"shapes_test_a_size{grid_size}.pl")
    with open(test_a_path, "rb") as file:
        test_a = dill.load(file)
        file.close()

    test_b_path = os.path.join(data_dir, f"shapes_test_b_size{grid_size}.pl")
    with open(test_b_path, "rb") as file:
        test_b = dill.load(file)
        file.close()

    val_a_path = os.path.join(data_dir, f"shapes_val_a_size{grid_size}.pl")
    with open(val_a_path, "rb") as file:
        val_a = dill.load(file)
        file.close()

    val_b_path = os.path.join(data_dir, f"shapes_val_b_size{grid_size}.pl")
    with open(val_b_path, "rb") as file:
        val_b = dill.load(file)
        file.close()

    anti_path = os.path.join(data_dir, f"shapes_anti_size{grid_size}.pl")
    with open(anti_path, "rb") as file:
        anti = dill.load(file)
        file.close()

    val_id_masks = compute_masks(X_val) if compute_mask else None
    val_dataset_id = data_cls(X_val, Y_val, masks=val_id_masks)

    val_a_masks = compute_masks(val_a["X_test_a"]) if compute_mask else None
    val_dataset_a = data_cls(val_a["X_test_a"], val_a["Y_test_a"], masks=val_a_masks)

    val_b_masks = compute_masks(val_b["X_test_b"]) if compute_mask else None
    val_dataset_b = data_cls(val_b["X_test_b"], val_b["Y_test_b"], masks=val_b_masks)

    test_id_masks = compute_masks(X_test) if compute_mask else None
    test_dataset_id = data_cls(X_test, Y_test, masks=test_id_masks)

    test_a_masks = compute_masks(test_a["X_test_a"]) if compute_mask else None
    test_dataset_a = data_cls(
        test_a["X_test_a"], test_a["Y_test_a"], masks=test_a_masks
    )

    test_b_masks = compute_masks(test_b["X_test_b"]) if compute_mask else None
    test_dataset_b = data_cls(
        test_b["X_test_b"], test_b["Y_test_b"], masks=test_b_masks
    )

    anti_masks = compute_masks(anti["X_anti"]) if compute_mask else None
    anti_dataset = data_cls(anti["X_anti"], anti["Y_anti"], masks=anti_masks)

    val_sets = [val_dataset_id, val_dataset_a, val_dataset_b]
    test_sets = [test_dataset_id, test_dataset_a, test_dataset_b]

    return dataset, val_sets, test_sets, anti_dataset


def get_boxworld_datasets(
    size: int, num_pairs: int, data_dir: str, one_hot: bool = False
):
    data_path = os.path.join(data_dir, f"boxworld_v2_train_pairs{num_pairs}.pl")
    data_path = to_absolute_path(data_path)
    data_cls = OneHotBoxWorld if one_hot else BasicDataset

    with open(data_path, "rb") as file:
        train_data = dill.load(file)
        file.close()

    midpoint = train_data["X_train"].size(0) // 2
    X_train_pos = train_data["X_train"][:midpoint]
    Y_train_pos = train_data["Y_train"][:midpoint]
    X_train_neg = train_data["X_train"][midpoint:]
    Y_train_neg = train_data["Y_train"][midpoint:]

    half_size = size // 2

    X_train = torch.cat([X_train_pos[:half_size], X_train_neg[:half_size]], dim=0)
    Y_train = torch.cat([Y_train_pos[:half_size], Y_train_neg[:half_size]], dim=0)

    X_val = torch.cat([X_train_pos[20000:20500], X_train_neg[20000:20500]], dim=0)
    Y_val = torch.cat([Y_train_pos[20000:20500], Y_train_neg[20000:20500]], dim=0)

    X_test = torch.cat([X_train_pos[20500:], X_train_neg[20500:]], dim=0)
    Y_test = torch.cat([Y_train_pos[20500:], Y_train_neg[20500:]], dim=0)

    dataset = data_cls(X_train, Y_train)

    test_col_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_test_col_pairs{num_pairs}.pl")
    )
    with open(test_col_path, "rb") as file:
        test_col = dill.load(file)
        file.close()

    val_col_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_val_col_pairs{num_pairs}.pl")
    )
    with open(val_col_path, "rb") as file:
        val_col = dill.load(file)
        file.close()

    test_pair_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_test_pair_pairs{num_pairs}.pl")
    )
    with open(test_pair_path, "rb") as file:
        test_pair = dill.load(file)
        file.close()

    val_pair_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_val_pair_pairs{num_pairs}.pl")
    )
    with open(val_pair_path, "rb") as file:
        val_pair = dill.load(file)
        file.close()

    test_dist_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_test_dist_pairs{num_pairs}.pl")
    )
    with open(test_dist_path, "rb") as file:
        test_dist = dill.load(file)
        file.close()

    val_dist_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_val_dist_pairs{num_pairs}.pl")
    )
    with open(val_dist_path, "rb") as file:
        val_dist = dill.load(file)
        file.close()

    test_comb_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_test_comb_pairs{num_pairs}.pl")
    )
    with open(test_comb_path, "rb") as file:
        test_comb = dill.load(file)
        file.close()

    val_comb_path = to_absolute_path(
        os.path.join(data_dir, f"boxworld_v2_val_comb_pairs{num_pairs}.pl")
    )
    with open(val_comb_path, "rb") as file:
        val_comb = dill.load(file)
        file.close()

    val_dataset_id = data_cls(X_val, Y_val)
    test_dataset_id = data_cls(X_test, Y_test)
    val_dataset_col = data_cls(val_col["X_col"], val_col["Y_col"])
    test_dataset_col = data_cls(test_col["X_col"], test_col["Y_col"])
    val_dataset_pair = data_cls(val_pair["X_pair"], val_pair["Y_pair"])
    test_dataset_pair = data_cls(test_pair["X_pair"], test_pair["Y_pair"])
    val_dataset_dist = data_cls(val_dist["X_dist"], val_dist["Y_dist"])
    test_dataset_dist = data_cls(test_dist["X_dist"], test_dist["Y_dist"])
    val_dataset_comb = data_cls(val_comb["X_comb"], val_comb["Y_comb"])
    test_dataset_comb = data_cls(test_comb["X_comb"], test_comb["Y_comb"])

    val_sets = [
        val_dataset_id,
        val_dataset_col,
        val_dataset_pair,
        val_dataset_dist,
        val_dataset_comb,
    ]
    test_sets = [
        test_dataset_id,
        test_dataset_col,
        test_dataset_pair,
        test_dataset_dist,
        test_dataset_comb,
    ]

    return dataset, val_sets, test_sets, None
