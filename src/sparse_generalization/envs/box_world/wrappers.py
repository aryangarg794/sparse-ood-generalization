import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from sparse_generalization.envs.box_world.env import BoxWorldEnv
gym.register('BoxWorldEnv-v1', BoxWorldEnv)

def make_env(
    num_pairs: int = 4, 
    unsolvable_prob: float = 0.0, 
    size: int = 15, 
    num_paths: int = 2, 
    include_walls: bool = False,
    num_distractors: int = 0,  
    render_mode: str = 'rgb_array',
    seed=None
):
    env = gym.make('BoxWorldEnv-v1', 
                   render_mode=render_mode, 
                   unsolvable_prob=unsolvable_prob, 
                   size=size, 
                   num_pairs=num_pairs,
                   include_walls=include_walls,
                   num_paths=num_paths,
                   num_distractors=num_distractors
                   )
    return ImgObsWrapper(FullyObsWrapper(env))