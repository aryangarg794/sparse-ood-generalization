import dill
import gymnasium as gym
import numpy as np
import argparse
import torch
from tqdm import tqdm

from sparse_generalization.data.box_world.env import BoxWorldEnvV2
from sparse_generalization.data.box_world.wrappers import make_env_v2

gym.register("BoxWorldEnv-v2", BoxWorldEnvV2)


def generate_dataset(
    num_samples: int = 1000,
    mode: str = "train",
    num_paths: int = 1,
    num_pairs: int = 2,
    file_path: str = "data/box_world/boxworld_v2",
):
    half_samples = num_samples // 2
    size = 10
    dataset = {}

    if mode == "train":
        print("Generating ID")
        env_solv = make_env_v2(
            size=size, num_pairs=num_pairs, ratio_red_blue=0.0, num_paths=num_paths
        )
        env_unsolv = make_env_v2(
            size=size, num_pairs=num_pairs, ratio_red_blue=1.0, num_paths=num_paths
        )

        X = []
        Y = []
        attn_edges = []

        for env, label, count in [
            (env_solv, 0, half_samples),
            (env_unsolv, 1, half_samples),
        ]:
            env.reset()
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

        dataset[f"X_{mode}"] = X
        dataset[f"Y_{mode}"] = Y
        dataset[f"edges_{mode}"] = attn_edges

    ### -------------- OOD: different colors ----------------
    elif mode == "col":
        print("Generating OOD Colors")

        X = []
        Y = []
        attn_edges = []

        env_solv_test = make_env_v2(
            size=size,
            num_pairs=num_pairs,
            ratio_red_blue=0.0,
            num_paths=num_paths,
            ood_colors=True,
        )
        env_unsolv_test = make_env_v2(
            size=size,
            num_pairs=num_pairs,
            ratio_red_blue=1.0,
            num_paths=num_paths,
            ood_colors=True,
        )

        for env, label, count in [
            (env_solv_test, 0, half_samples),
            (env_unsolv_test, 1, half_samples),
        ]:
            env.reset()
            for _ in tqdm(range(count), desc=f"Generating {label} OOD colors"):
                obs, _ = env.reset()
                edges = env.get_wrapper_attr("get_attn_edges")()
                X.append(obs)
                Y.append(label)
                attn_edges.append(edges)
            env.close()

        X = torch.from_numpy(np.array(X))
        Y = torch.from_numpy(np.array(Y))

        dataset[f"X_{mode}"] = X
        dataset[f"Y_{mode}"] = Y
        torch_edges = []
        for edges_arr in tqdm(attn_edges, desc="Converting edges"):
            torch_edges.append(torch.from_numpy(np.array(edges_arr)))
        dataset[f"edges_{mode}"] = torch_edges

    ### --------------- OOD: different num pairs ------------
    elif mode == "pair":
        print("Generating OOD Pairs")
        X = []
        Y = []
        attn_edges = []

        env_solv_test = make_env_v2(
            size=size, num_pairs=-1, ratio_red_blue=0.0, num_paths=num_paths
        )
        env_unsolv_test = make_env_v2(
            size=size, num_pairs=-1, ratio_red_blue=1.0, num_paths=num_paths
        )

        for env, label, count in [
            (env_solv_test, 0, half_samples),
            (env_unsolv_test, 1, half_samples),
        ]:
            env.reset()
            for _ in tqdm(range(count), desc=f"Generating {label} OOD pairs"):
                obs, _ = env.reset()
                edges = env.get_wrapper_attr("get_attn_edges")()
                X.append(obs)
                Y.append(label)
                attn_edges.append(edges)
            env.close()

        X = torch.from_numpy(np.array(X))
        Y = torch.from_numpy(np.array(Y))

        dataset[f"X_{mode}"] = X
        dataset[f"Y_{mode}"] = Y
        torch_edges = []
        for edges_arr in tqdm(attn_edges, desc="Converting edges"):
            torch_edges.append(torch.from_numpy(np.array(edges_arr)))
        dataset[f"edges_{mode}"] = torch_edges

    ### ------------------- OOD: distractors ---------------
    elif mode == "dist":
        print("Generating OOD Dist")

        X = []
        Y = []
        attn_edges = []

        env_solv_test = make_env_v2(
            size=size,
            num_pairs=num_pairs,
            ratio_red_blue=0.0,
            num_paths=num_paths,
            num_distractors=5,
        )
        env_unsolv_test = make_env_v2(
            size=size,
            num_pairs=num_pairs,
            ratio_red_blue=1.0,
            num_paths=num_paths,
            num_distractors=5,
        )

        for env, label, count in [
            (env_solv_test, 0, half_samples),
            (env_unsolv_test, 1, half_samples),
        ]:
            env.reset()
            for _ in tqdm(range(count), desc=f"Generating {label} OOD distractors"):
                obs, _ = env.reset()
                edges = env.get_wrapper_attr("get_attn_edges")()
                X.append(obs)
                Y.append(label)
                attn_edges.append(edges)
            env.close()

        X = torch.from_numpy(np.array(X))
        Y = torch.from_numpy(np.array(Y))

        dataset[f"X_{mode}"] = X
        dataset[f"Y_{mode}"] = Y

        torch_edges = []
        for edges_arr in tqdm(attn_edges, desc="Converting edges"):
            torch_edges.append(torch.from_numpy(np.array(edges_arr)))
        dataset[f"edges_{mode}"] = torch_edges

    ### ------------------- OOD: combined ---------------
    elif mode == "comb":
        print("Generating Combined")

        X = []
        Y = []
        attn_edges = []

        env_solv_test = make_env_v2(
            size=size,
            num_pairs=-1,
            ratio_red_blue=0.0,
            num_paths=num_paths,
            num_distractors=5,
            ood_colors=True,
        )
        env_unsolv_test = make_env_v2(
            size=size,
            num_pairs=-1,
            ratio_red_blue=1.0,
            num_paths=num_paths,
            num_distractors=5,
            ood_colors=True,
        )

        for env, label, count in [
            (env_solv_test, 0, half_samples),
            (env_unsolv_test, 1, half_samples),
        ]:
            env.reset()
            for _ in tqdm(range(count), desc=f"Generating {label} OOD combined"):
                obs, _ = env.reset()
                edges = env.get_wrapper_attr("get_attn_edges")()
                X.append(obs)
                Y.append(label)
                attn_edges.append(edges)
            env.close()

        X = torch.from_numpy(np.array(X))
        Y = torch.from_numpy(np.array(Y))

        dataset[f"X_{mode}"] = X
        dataset[f"Y_{mode}"] = Y
        torch_edges = []
        for edges_arr in tqdm(attn_edges, desc="Converting edges"):
            torch_edges.append(torch.from_numpy(np.array(edges_arr)))
        dataset[f"edges_{mode}"] = torch_edges

    print("Saving")
    file_path = file_path + f"_{mode}_{num_samples}_pairs{num_pairs}.pl"
    with open(file_path, "wb") as f:
        dill.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num_samples", type=int, default=250, help="numnber of samples"
    )
    parser.add_argument("-p", "--num_pairs", type=int, default=2, help="num of pairs")
    parser.add_argument("-pt", "--num_paths", type=int, default=1, help="num of paths")
    parser.add_argument("-m", "--mode", type=str, default="train", help="mode name")

    args = parser.parse_args()

    generate_dataset(
        num_pairs=args.num_pairs,
        num_samples=args.num_samples,
        mode=args.mode,
        num_paths=args.num_paths,
    )
