import dill
import gymnasium as gym
import numpy as np 
import random
import torch

from sparse_generalization.envs.box_world.env import BoxWorldEnv
from sparse_generalization.envs.box_world.wrappers import make_env

gym.register('BoxWorldEnv-v1', BoxWorldEnv)

def generate_dataset(
    num_samples: int = 500, 
    ratio_labels: float = 0.5,
    test_size: float = 2.0,  
    start_seed: int = 0,
    num_paths: int = 1, 
    file_path: str = 'data/box_world/balanced_dist' 
):  
    num_test = int(num_samples * test_size) # 1000
    num_total = num_samples + num_test # 1500
    num_solv = int(num_total * ratio_labels) # 750 
    num_unsolv = num_total - num_solv # 750
    
    env_solv = make_env(size=10, num_pairs=3, unsolvable_prob=0.0, num_paths=num_paths)
    
    X = []
    Y = []
    attn_edges = []
    env_solv.reset(seed=start_seed)
    
    for _ in range(num_solv):
        obs, _ = env_solv.reset()
        edges = env_solv.get_wrapper_attr('get_attn_edges')()
        X.append(obs)
        Y.append(1)
        attn_edges.append(edges)

    env_unsolv = make_env(size=10, num_pairs=3, unsolvable_prob=1.0, num_paths=num_paths)
    env_unsolv.reset(seed=start_seed)
        
    for _ in range(num_unsolv):
        obs, _ = env_unsolv.reset()
        edges = env_unsolv.get_wrapper_attr('get_attn_edges')()
        X.append(obs)
        Y.append(0)
        attn_edges.append(edges)
    
    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    attn_edges = torch.from_numpy(np.array(attn_edges))
    perm = torch.randperm(num_total)
    train_idx = perm[:num_samples]
    test_idx = perm[num_samples:]
    
    dataset = {
        "X_train": X[train_idx],
        "Y_train": Y[train_idx],
        "X_test_ind": X[test_idx],
        "Y_test_ind": Y[test_idx],
        "edges_train": attn_edges[train_idx],
        "edges_test_ind": attn_edges[test_idx]
    }
    
    num_solv_test = int(num_test * ratio_labels)
    num_unsolv_test = num_test - num_solv_test
    
    X = []
    Y = []
    attn_edges = []
    
    env_solv_test = make_env(size=10, num_pairs=-1, unsolvable_prob=0.0, num_paths=num_paths, num_distractors=4)
    env_solv_test.reset(seed=start_seed)
    
    for _ in range(num_solv_test):
        obs, _ = env_solv_test.reset()
        edges = env_solv_test.get_wrapper_attr('get_attn_edges')()
        X.append(obs)
        Y.append(1)
        attn_edges.append(edges)
        
    env_unsolv_test = make_env(size=10, num_pairs=-1, unsolvable_prob=1.0, num_paths=num_paths, num_distractors=4)
    env_unsolv_test.reset(seed=start_seed)
    
    for _ in range(num_unsolv_test):
        obs, _ = env_unsolv_test.reset()
        edges = env_unsolv_test.get_wrapper_attr('get_attn_edges')()
        X.append(obs)
        Y.append(0)
        attn_edges.append(edges)
    
    
    dataset['X_test_ood'] = torch.from_numpy(np.array(X))
    dataset['Y_test_ood'] = torch.from_numpy(np.array(Y))
    torch_edges = []
    for edges_arr in attn_edges:
        torch_edges.append(torch.from_numpy(np.array(edges_arr)))
    dataset['edges_test_ood'] = torch_edges
    
    file_path = file_path + f'_{num_samples}.pl'
    with open(file_path, 'wb') as f:
        dill.dump(dataset, f)

if __name__ == "__main__":
    generate_dataset()