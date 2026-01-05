import dill
import gymnasium as gym
import numpy as np 
import random
import torch

from sparse_generalization.envs.box_world.env import BoxWorldEnv
from sparse_generalization.envs.box_world.wrappers import make_env

gym.register('BoxWorldEnv-v1', BoxWorldEnv)

def generate_dataset(
    num_samples: int = 50000, 
    ratio_labels: float = 0.5, 
    seed: int | None = None,
    file_path: str = 'data/box_world/balanced_test' 
):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
        
    num_solv = int(num_samples * ratio_labels)
    num_unsolv = num_samples - num_solv
    
    env_solv = make_env(include_walls=False, size=8, num_pairs=3, unsolvable_prob=0.0)
    
    X = []
    Y = []
    
    for _ in range(num_solv):
        obs, _ = env_solv.reset()
        X.append(obs)
        Y.append(1)
        
    env_unsolv = make_env(include_walls=False, size=8, num_pairs=3, unsolvable_prob=1.0)
        
    for _ in range(num_unsolv):
        obs, _ = env_unsolv.reset()
        X.append(obs)
        Y.append(0)
    
    dataset = {"X": torch.from_numpy(np.array(X)), "Y": torch.from_numpy(np.array(Y))}
    
    file_path = file_path + f'_{num_samples}.pl'
    with open(file_path, 'wb') as f:
        dill.dump(dataset, f)

if __name__ == "__main__":
    generate_dataset()