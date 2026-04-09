import dill
import gymnasium as gym
import numpy as np
import random
import torch
from tqdm import tqdm

from sparse_generalization.data.box_world.env import BoxWorldEnv
from sparse_generalization.data.box_world.wrappers import make_env

gym.register("BoxWorldEnv-v1", BoxWorldEnv)


def generate_dataset(
    num_samples: int = 50000,
    ratio_labels: float = 0.5,
    val_size: int = 1000,
    test_size: int = 5000,
    start_seed: int = 0,
    num_paths: int = 1,
    render: bool = False,
    file_path: str = "data/box_world/balanced_ood",
):
    print("Generating ID")
    num_test = test_size
    num_total = num_samples + num_test + val_size
    num_solv = int(num_total * ratio_labels)
    num_unsolv = num_total - num_solv

    env_solv = make_env(size=10, num_pairs=3, unsolvable_prob=0.0, num_paths=num_paths)
    env_unsolv = make_env(
        size=10, num_pairs=3, unsolvable_prob=1.0, num_paths=num_paths
    )

    X = []
    Y = []
    attn_edges = []

    for env, label, count in [(env_solv, 1, num_solv), (env_unsolv, 0, num_unsolv)]:
        env.reset(seed=start_seed)
        for _ in tqdm(range(count), desc=f"Generating {label} samples"):
            obs, _ = env.reset()
            edges = env.get_wrapper_attr("get_attn_edges")()
            X.append(obs)
            Y.append(label)
            attn_edges.append(edges)
        env.close()

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    attn_edges = torch.from_numpy(np.array(attn_edges))
    perm = torch.randperm(num_total)
    train_idx = perm[:num_samples]
    val_idx = perm[num_samples : (val_size + num_samples)]
    test_idx = perm[(val_size + num_samples) :]

    dataset = {
        "X_train": X[train_idx],
        "Y_train": Y[train_idx],
        "X_val_id": X[val_idx],
        "Y_val_id": Y[val_idx],
        "X_test_id": X[test_idx],
        "Y_test_id": Y[test_idx],
        "edges_train": attn_edges[train_idx],
        "edges_val_id": attn_edges[val_idx],
        "edges_test_id": attn_edges[test_idx],
    }

    ### -------------- OOD: different colors ----------------
    print("Generating OOD Colors")
    num_test_total = num_test + val_size
    num_solv_test = int(num_test_total * ratio_labels)
    num_unsolv_test = num_test_total - num_solv_test

    X = []
    Y = []
    attn_edges = []

    env_solv_test = make_env(
        size=10, num_pairs=3, unsolvable_prob=0.0, num_paths=num_paths, ood_colors=True
    )
    env_unsolv_test = make_env(
        size=10, num_pairs=3, unsolvable_prob=1.0, num_paths=num_paths, ood_colors=True
    )

    for env, label, count in [
        (env_solv_test, 1, num_solv_test),
        (env_unsolv_test, 0, num_unsolv_test),
    ]:
        env.reset(seed=start_seed)
        for _ in tqdm(range(count), desc=f"Generating {label} OOD colors"):
            obs, _ = env.reset()
            edges = env.get_wrapper_attr("get_attn_edges")()
            X.append(obs)
            Y.append(label)
            attn_edges.append(edges)
        env.close()

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    perm = torch.randperm(num_test_total)
    val_idx = perm[:val_size]
    test_idx = perm[val_size:]

    dataset["X_val_col"] = X[val_idx]
    dataset["Y_val_col"] = Y[val_idx]
    dataset["X_test_col"] = X[test_idx]
    dataset["Y_test_col"] = Y[test_idx]
    torch_edges = []
    for edges_arr in tqdm(attn_edges, desc="Converting edges"):
        torch_edges.append(torch.from_numpy(np.array(edges_arr)))
    dataset["edges_test_col"] = torch_edges

    ### --------------- OOD: different num pairs ------------
    print("Generating OOD Pairs")
    num_test_total = num_test + val_size
    num_solv_test = int(num_test_total * ratio_labels)
    num_unsolv_test = num_test_total - num_solv_test

    X = []
    Y = []
    attn_edges = []

    env_solv_test = make_env(
        size=10, num_pairs=-1, unsolvable_prob=0.0, num_paths=num_paths
    )
    env_unsolv_test = make_env(
        size=10, num_pairs=-1, unsolvable_prob=1.0, num_paths=num_paths
    )

    for env, label, count in [
        (env_solv_test, 1, num_solv_test),
        (env_unsolv_test, 0, num_unsolv_test),
    ]:
        env.reset(seed=start_seed)
        for _ in tqdm(range(count), desc=f"Generating {label} OOD pairs"):
            obs, _ = env.reset()
            edges = env.get_wrapper_attr("get_attn_edges")()
            X.append(obs)
            Y.append(label)
            attn_edges.append(edges)
        env.close()

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    perm = torch.randperm(num_test_total)
    val_idx = perm[:val_size]
    test_idx = perm[val_size:]

    dataset["X_val_pair"] = X[val_idx]
    dataset["Y_val_pair"] = Y[val_idx]
    dataset["X_test_pair"] = X[test_idx]
    dataset["Y_test_pair"] = Y[test_idx]
    torch_edges = []
    for edges_arr in tqdm(attn_edges, desc="Converting edges"):
        torch_edges.append(torch.from_numpy(np.array(edges_arr)))
    dataset["edges_test_pair"] = torch_edges

    ### ------------------- OOD: distractors ---------------
    print("Generating OOD Dist")
    num_test_total = num_test + val_size
    num_solv_test = int(num_test_total * ratio_labels)
    num_unsolv_test = num_test_total - num_solv_test

    X = []
    Y = []
    attn_edges = []

    env_solv_test = make_env(
        size=10,
        num_pairs=3,
        unsolvable_prob=0.0,
        num_paths=num_paths,
        num_distractors=5,
    )
    env_unsolv_test = make_env(
        size=10,
        num_pairs=3,
        unsolvable_prob=1.0,
        num_paths=num_paths,
        num_distractors=5,
    )

    for env, label, count in [
        (env_solv_test, 1, num_solv_test),
        (env_unsolv_test, 0, num_unsolv_test),
    ]:
        env.reset(seed=start_seed)
        for _ in tqdm(range(count), desc=f"Generating {label} OOD distractors"):
            obs, _ = env.reset()
            edges = env.get_wrapper_attr("get_attn_edges")()
            X.append(obs)
            Y.append(label)
            attn_edges.append(edges)
        env.close()

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    perm = torch.randperm(num_test_total)
    val_idx = perm[:val_size]
    test_idx = perm[val_size:]

    dataset["X_val_dist"] = X[val_idx]
    dataset["Y_val_dist"] = Y[val_idx]
    dataset["X_test_dist"] = X[test_idx]
    dataset["Y_test_dist"] = Y[test_idx]
    torch_edges = []
    for edges_arr in tqdm(attn_edges, desc="Converting edges"):
        torch_edges.append(torch.from_numpy(np.array(edges_arr)))
    dataset["edges_test_dist"] = torch_edges

    ### ------------------- OOD: combined ---------------
    print("Generating Combined")
    num_test_total = num_test + val_size
    num_solv_test = int(num_test_total * ratio_labels)
    num_unsolv_test = num_test_total - num_solv_test

    X = []
    Y = []
    attn_edges = []

    env_solv_test = make_env(
        size=10,
        num_pairs=-1,
        unsolvable_prob=0.0,
        num_paths=num_paths,
        num_distractors=5,
        ood_colors=True,
    )
    env_unsolv_test = make_env(
        size=10,
        num_pairs=-1,
        unsolvable_prob=1.0,
        num_paths=num_paths,
        num_distractors=5,
        ood_colors=True,
    )

    for env, label, count in [
        (env_solv_test, 1, num_solv_test),
        (env_unsolv_test, 0, num_unsolv_test),
    ]:
        env.reset(seed=start_seed)
        for _ in tqdm(range(count), desc=f"Generating {label} OOD combined"):
            obs, _ = env.reset()
            edges = env.get_wrapper_attr("get_attn_edges")()
            X.append(obs)
            Y.append(label)
            attn_edges.append(edges)
        env.close()

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    perm = torch.randperm(num_test_total)
    val_idx = perm[:val_size]
    test_idx = perm[val_size:]

    dataset["X_val_comb"] = X[val_idx]
    dataset["Y_val_comb"] = Y[val_idx]
    dataset["X_test_comb"] = X[test_idx]
    dataset["Y_test_comb"] = Y[test_idx]
    torch_edges = []
    for edges_arr in tqdm(attn_edges, desc="Converting edges"):
        torch_edges.append(torch.from_numpy(np.array(edges_arr)))
    dataset["edges_test_comb"] = torch_edges

    print("Saving")
    file_path = file_path + f"_{num_samples}.pl"
    with open(file_path, "wb") as f:
        dill.dump(dataset, f)


if __name__ == "__main__":
    generate_dataset()
