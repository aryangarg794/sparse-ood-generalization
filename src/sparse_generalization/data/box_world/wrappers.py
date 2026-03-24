import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from sparse_generalization.data.box_world.env import BoxWorldEnv, BoxWorldEnvV2
gym.register('BoxWorldEnv-v1', BoxWorldEnv)
gym.register('BoxWorldEnv-v2', BoxWorldEnvV2)

def make_env(
    num_pairs: int = 4, 
    unsolvable_prob: float = 0.0, 
    size: int = 15, 
    num_paths: int = 2, 
    include_walls: bool = False,
    num_distractors: int = 0,  
    ood_colors: bool = False, 
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
                   num_distractors=num_distractors,
                   ood_colors=ood_colors
                   )
    return ImgObsWrapper(FullyObsWrapper(env))


def make_env_v2(
    num_pairs: int = 4, 
    ratio_red_blue: float = 0.5, 
    size: int = 15, 
    num_paths: int = 2, 
    include_walls: bool = False,
    num_distractors: int = 0,  
    ood_colors: bool = False, 
    render_mode: str = 'rgb_array',
    seed=None
):
    env = gym.make('BoxWorldEnv-v2', 
                   render_mode=render_mode, 
                   ratio_red_blue=ratio_red_blue, 
                   size=size, 
                   num_pairs=num_pairs,
                   include_walls=include_walls,
                   num_paths=num_paths,
                   num_distractors=num_distractors,
                   ood_colors=ood_colors
                   )
    return ImgObsWrapper(FullyObsWrapper(env))