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
    seed: int = 0,
    num_paths: int = 1, 
    file_path: str = 'data/box_world/balanced_dist' 
):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    num_test = int(num_samples * test_size)
    num_total = num_samples + num_test
    num_solv = int((num_samples+num_test) * ratio_labels)
    num_unsolv = (num_samples+num_test) - num_solv
    
    env_solv = make_env(size=10, num_pairs=3, unsolvable_prob=0.0, num_paths=num_paths)
    
    X = []
    Y = []
    
    for _ in range(num_solv):
        obs, _ = env_solv.reset(seed=seed)
        X.append(obs)
        Y.append(1)
        
    env_unsolv = make_env(size=10, num_pairs=3, unsolvable_prob=1.0, num_paths=num_paths)
        
    for _ in range(num_unsolv):
        obs, _ = env_unsolv.reset(seed=seed)
        X.append(obs)
        Y.append(0)
    
    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    perm = torch.randperm(num_total)
    train_idx = perm[:num_test]
    test_idx = perm[num_test:]
    
    dataset = {
        "X_train": X[train_idx],
        "Y_train": Y[train_idx],
        "X_test_ind": X[test_idx],
        "Y_test_ind": Y[test_idx],
    }
    
    num_solv_test = int(num_test * ratio_labels)
    num_unsolv_test = num_test - num_solv_test
    
    X = []
    Y = []
    
    env_solv_test = make_env(size=10, num_pairs=-1, unsolvable_prob=0.0, num_paths=num_paths, num_distractors=3)
    
    for _ in range(num_solv_test):
        obs, _ = env_solv_test.reset(seed=seed)
        X.append(obs)
        Y.append(1)
        
    env_unsolv_test = make_env(size=10, num_pairs=-1, unsolvable_prob=1.0, num_paths=num_paths, num_distractors=3)
    
    for _ in range(num_unsolv_test):
        obs, _ = env_unsolv_test.reset(seed=seed)
        X.append(obs)
        Y.append(0)
    
    dataset['X_test_ood'] = torch.from_numpy(np.array(X))
    dataset['Y_test_ood'] = torch.from_numpy(np.array(Y))
    
    file_path = file_path + f'_{num_samples}.pl'
    with open(file_path, 'wb') as f:
        dill.dump(dataset, f)

if __name__ == "__main__":
    generate_dataset()