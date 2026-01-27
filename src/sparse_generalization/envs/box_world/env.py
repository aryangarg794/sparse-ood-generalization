import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import random

from copy import deepcopy
from collections import defaultdict
from itertools import product
from PIL import Image
from minigrid.minigrid_env import MiniGridEnv 
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid

from sparse_generalization.envs.box_world.objects import Wall, Goal, KeyBox, LockBox
from sparse_generalization.envs.box_world.constants import COLOR_NAMES

# somewhat inspired from https://github.com/aryangarg794/rnd_dqn_four_room/blob/master/four_room/env.py
class BoxWorldEnv(MiniGridEnv):
    """Simple crossing based environment. Currently only for visual purposes, movement
    and transition logic still needs to be added.

    Args:
        agent_pos (list): List of contextual agent starting positions.
        agent_dir (list): List of starting agent directions.
        goal_pos (list): List of goal positions.
        topologies (list): List of lists of walls in the environment. 
        key_topologies (list): List of lists of tuples. The first tuple is on its own to indicate the first key, \
            the second tuple is another tuple which indicates key-lock pairs.
    """
    
    def __init__(
        self, 
        agent_pos=None, 
        agent_dir=None, 
        goal_pos=None, 
        topologies=None,
        key_topologies=None,  
        size=12, 
        num_pairs=4, 
        num_distractors=0,
        max_steps=100,
        num_paths=2, 
        highlight=False,
        include_walls=False,
        unsolvable_prob=0.0,
        edge_remove_prob=0.3,
        max_tries=50, 
        render_mode='human',
    ):
        self._agent_pos_list = agent_pos
        self._goal_pos_list = goal_pos
        self._topology_list = topologies
        self._agent_dir_list = agent_dir
        self._key_topology_list = key_topologies
        
        self._randomize = True if (self._agent_dir_list is None and 
                                   self._goal_pos_list is None and 
                                   self._agent_pos_list is None and 
                                   self._topology_list is None and
                                   self._key_topology_list is None) else False
        
        self._list_idx = None if self._randomize else 0
        self._current_context = 0
        self.num_distractors = num_distractors
        self.unsolvable_prob = unsolvable_prob
        self.num_pairs = num_pairs
        self.edge_drop_prob = edge_remove_prob
        self._include_walls = include_walls
        self.goal_pos = None
        
        if not self._randomize:
            self._list_size = len(self._agent_pos_list)
            assert len(self._agent_dir_list) == self._list_size
            assert len(self._goal_pos_list) == self._list_size
            assert len(self._walls_pos_list) == self._list_size
        
        mission = MissionSpace(mission_func=lambda: "get the keys and reach the goal")
        
        super().__init__(
            mission_space=mission,
            width=size,
            height=size,
            max_steps=max_steps, 
            highlight=highlight,
            render_mode=render_mode
        )  
        
        self.size = size
        self.num_paths = num_paths
        self.paths = {}
        self.path_colors = {}
        self.edges = {}
        self.max_tries = max_tries
    
    def _gen_grid(self, width, height):
        self._current_context = self._list_idx if not self._randomize else None
        self.grid = Grid(width, height)
    
        self.grid.wall_rect(0, 0, width, height)
        
        if self._randomize:
            # waalls: for now simple just one wall
            if self._include_walls:
                self.walls = self._simple_wall_gen()
                for wall in self.walls:
                    self.grid.set(*wall, Wall())
        
            # agent and goal logic
            self.place_agent()
            free = False
            while not free:
                self.goal_pos = random.choice(self.valid_pos_gen(margin=1))
                next_goal = self._sanitize((self.goal_pos[0]+1, self.goal_pos[1]))
                if self.grid.get(*next_goal) is None:
                    free = True
                
            self.grid.set(*self.goal_pos, Goal())
            
            for _ in range(self.max_tries):
                try:
                    colors, last_color = self._gen_grid_once()
                    break
                except IndexError:
                    continue                         
                    
            if np.random.rand() < self.unsolvable_prob:
                lowest_idx = 10000
                
                for path in range(self.num_paths):
                    idx = random.choice(list(range(1, self.num_pairs)))
                    multiplier = path * (self.num_pairs) 
                    for i in range(1, self.num_pairs):
                        rand_color = random.choice(colors)
                        if i == idx:
                            key_lock = self.paths[path][i]
                            key, lock = key_lock
                            self.grid.set(*key, KeyBox(rand_color, index=idx, path=0))
                            self.edges[path].remove((i-1+multiplier, i+multiplier))
                            lowest_idx = min(lowest_idx, i)
            
                        elif np.random.rand() < self.edge_drop_prob:
                            key_lock = self.paths[path][i]
                            key, lock = key_lock
                            self.grid.set(*key, KeyBox(rand_color, index=idx, path=0))
                            self.edges[path].remove((i-1+multiplier, i+multiplier))
                            lowest_idx = min(lowest_idx, i)
                            
                    self.paths[path] = self.paths[path][0:lowest_idx]
        
            self.grid.set(self.goal_pos[0]+1, self.goal_pos[1], LockBox(last_color, goal_lock=True, index=self.num_pairs))
                
            
        else: #TODO: from given contexts
            self.walls = self._topology_list[self._list_idx]
            self.agent_pos = self._agent_pos_list[self._list_idx]
            self.agent_dir = self._agent_dir_list[self._list_idx]
            self.keys = self._key_topology_list[self._list_idx]
            self.goal_pos = self._goal_pos_list[self._list_idx]
            
        self.unlocked_edges = []
        
        # step logic
        self._current_key = None
        self._current_path = None
        
        
    def _gen_grid_once(self):
        last_color = None
        for path in range(self.num_paths):
            samples = self.num_pairs
            colors = deepcopy(COLOR_NAMES)
            for color in ['green', 'grey', 'red']: # base objs
                colors.remove(color)
            pairs, colors_path = self._sample_key_pairs(samples, colors)
                
            for color in colors_path: # no clash with colors
                colors.remove(color)
                
            # all end colors should be the same
            if path > 0:
                colors_path[-1] = last_color
            else:
                last_color = colors_path[-1]
            
            multiplier = path * (self.num_pairs)
            edges = []
            self.grid.set(*pairs[0], KeyBox(colors_path[0], index=0, first_key=True, path=path))
            for i, key_lock in enumerate(pairs[1:]):
                idx = i+1
                lock, key = key_lock
                self.grid.set(*key, KeyBox(colors_path[idx], index=idx, path=0))
                self.grid.set(*lock, LockBox(colors_path[idx-1], index=idx, path=0))
                edges.append((i+multiplier, idx+multiplier))
                
            pairs.append(((self.goal_pos[0]+1, self.goal_pos[1]), self.goal_pos))
            
            self.paths[path] = pairs
            self.path_colors[path] = colors_path
            self.edges[path] = edges
            
        return colors, last_color
    
    def reset(self, *, seed=None, options=None):
        if seed and not self._randomize:
            random.seed(seed)
            self._list_idx = random.randint(0, self._list_size-1)
        else:
            pass
            
        return super().reset(seed=seed, options=options)    
        
    def step(self, action):
        self.step_count += 1 
        
        reward = 0
        terminated = False
        truncated = False
        
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        cur_cell = self.grid.get(*self.agent_pos)

        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if (fwd_cell is not None and fwd_cell.type == "goal" and 
                self._current_key == self.num_pairs and self._current_path == fwd_cell.path and fwd_cell.unlockable):
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # pick up an object (this action is like interact for this env)
        elif action == self.actions.pickup:
            if cur_cell is not None and cur_cell.can_pickup():
                if self._current_key is None and cur_cell.is_first_key(): # first key
                    self._current_key = cur_cell.index
                    self._current_path = cur_cell.path
                    
                    next_lock, _  = self.pairs[cur_cell.path][cur_cell.index+1] # we can open the next lock
                    lock_cell = self.grid.get(*next_lock)
                    assert lock_cell.is_lock(), 'Not a lock'
                    lock_cell.unlockable = True
                    
                    # remove the key from the grid
                    self.grid.set(*self.agent_pos, None)
                    
                    # make the other path unusable
                    other_key_pos = self.pairs[1-cur_cell.path][cur_cell.index]
                    other_key = self.grid.get(*other_key_pos)
                    other_key.interactable = False
                    
                    # make the goal reachable 
                    goal_cell = self.grid.get(self.goal_pos[0]+1, self.goal_pos[1])
                    goal_cell.path = cur_cell.path
                        
                # opening a lock for which we have the key    
                elif self._current_key is not None and cur_cell.path == self._current_path and cur_cell.is_lock(): 
                    # first do the logic for if the lock is next to goal                    
                    self._current_key = cur_cell.index
                    self.unlocked_edges.append((cur_cell.index-1, cur_cell.index) \
                        if cur_cell.path == 0 else (cur_cell.index-1+self.num_pairs, cur_cell.index+self.num_pairs))
                    if cur_cell.is_goal_lock():
                        self.grid.set(*self.agent_pos, None)
                    else:
                        next_lock, _ = self.pairs[cur_cell.path][cur_cell.index+1]
                        lock_cell = self.grid.get(*next_lock)
                        assert lock_cell.is_lock() or lock_cell.is_goal(), 'Not a lock or goal'
                        lock_cell.unlockable = True
                        
                        if self._current_key == self.num_pairs-1:
                            _, goal_pos = self.pairs[cur_cell.path][cur_cell.index+1]
                            goal_cell = self.grid.get(*goal_pos)
                            goal_cell.unlockable = True
                            
                        # remove current key-lock 
                        cur_key, cur_lock = self.pairs[cur_cell.path][cur_cell.index]
                        self.grid.set(*cur_key, None)
                        self.grid.set(*cur_lock, None)
 

        # Done action (not used by default)
        elif action == self.actions.done:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        
        obs = self.gen_obs()
        
        return obs, reward, terminated, truncated, {}
        
    def _simple_wall_gen(self, min_size=2):
        walls = []
        orient = random.randint(0, 1)
        place = random.choice(list(range(2+min_size, self.size-2-min_size))) # at least 1 space at start, or 2 atr end
        door_size = random.randint(1, self.size-4)
        door_place = random.randint(1, self.size-1-door_size)
        for i in range(1, self.size):
            if door_place <= i <= door_place+door_size:
                continue
            wall = (place, i) if orient == 1 else (i, place)
            walls.append(wall)
            
        return walls

    
    def _sample_key_pairs(self, num_pairs, colors_list):
        pairs = []
        valid_pos = self.valid_pos_gen()
        free = False
        while not free:
            first_key = random.choice(valid_pos)
            right_state = self._sanitize((first_key[0]+1, first_key[1]))
            left_state = self._sanitize((first_key[0]-1, first_key[1]))
            if self.grid.get(*right_state) is None and self.grid.get(*left_state) is None:
                free = True
        pairs.append(first_key)
        valid_pos.remove(first_key)
        # remove states on left and right of first key
        state_right = self._sanitize((first_key[0]+1, first_key[1]))
        state_left = self._sanitize((first_key[0]-1, first_key[1]))
        if state_right in valid_pos: valid_pos.remove(state_right)
        if state_left in valid_pos: valid_pos.remove(state_left)
        
        while len(pairs) < num_pairs:
            lock = random.choice(valid_pos)
            valid_pos.remove(self._sanitize(lock))
            if lock[0] == 1 or np.array_equal((lock[0]-1, lock[1]), self.agent_pos) \
                or self.grid.get(lock[0]-1, lock[1]) is not None: 
                continue
            key = (lock[0]-1, lock[1])
            key_sanitized = self._sanitize(key)
            if key_sanitized not in valid_pos: continue
            valid_pos.remove(key_sanitized)       
            pairs.append((lock, key))
        
        colors = random.sample(colors_list, k=num_pairs)
            
        return pairs, colors
    
    def valid_pos_gen(self, margin=1):
        valid_pos_gen = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj is None:  
                    if (margin <= x <= self.width - margin-1 and 
                        margin <= y <= self.height - margin-1) :
                        valid_pos_gen.append((x, y))

        sanitized_pos = self._sanitize(self.agent_pos)
        if sanitized_pos in valid_pos_gen:
            valid_pos_gen.remove(sanitized_pos)
        
        # check other path key-locks
        for _, path in self.paths.items():
            first = path[0]
            first_left = self._sanitize((first[0]-1, first[1]))
            first_right = self._sanitize((first[0]+1, first[1]))
            if first_left in valid_pos_gen: valid_pos_gen.remove(first_left)
            if first_right in valid_pos_gen: valid_pos_gen.remove(first_right)
            
            for pair in range(1, self.num_pairs):
                key_lock = path[pair]
                left = self._sanitize((key_lock[1][0]-1, key_lock[1][1]))
                right = self._sanitize((key_lock[0][0]+1, key_lock[0][1]))
                if left in valid_pos_gen: valid_pos_gen.remove(left)
                if right in valid_pos_gen: valid_pos_gen.remove(right)
                
        # goal state stuff
        if self.goal_pos:
            for i in range(1, 4):
                state = self._sanitize((self.goal_pos[0]+i, self.goal_pos[1]))
                if state in valid_pos_gen: valid_pos_gen.remove(state)
        
        return valid_pos_gen
    
    def set_agent(self, pos, dir=0):
        self.agent_pos = pos
        self.agent_dir = dir
    
    def render_graph(self, figsize=(10, 6)):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        img = self.render()
        ax[0].imshow(img)
        
        def custom_layout(G, num_pairs, n_paths,
                  y_top=0.0, y_spacing=0.5, x_spacing=0.5):
            pos = {}
            y_values = [y_top - i * y_spacing for i in range(n_paths)]
            for path_idx, y in enumerate(y_values):
                for i in range(num_pairs):
                    node_id = path_idx * num_pairs + i
                    pos[node_id] = (i * x_spacing, y)
            center_x = num_pairs * x_spacing
            y_mid = sum(y_values) / len(y_values)
            pos[n_paths * num_pairs] = (center_x, y_mid)

            return pos
        
        # draw graph 
        edges = []
        for path in range(self.num_paths):
            edges += self.edges[path]
            
        colors = []
        for path in range(self.num_paths):
            colors += self.path_colors[path]
        colors = colors + ['springgreen']
        nodes = [i for i in range(len(colors))] 
        nodes = nodes + [self.num_pairs*self.num_paths]
        edges = edges + [(self.num_pairs*(path+1)-1, self.num_pairs*self.num_paths) for path in range(self.num_paths)]
        
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        pos = custom_layout(G, self.num_pairs, self.num_paths)
        edge_colors = ['red' if e in self.unlocked_edges else 'black' for e in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_color=colors, ax=ax[1], node_size=1000, linewidths=1, edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=ax[1])
        
        fig.tight_layout()
        plt.axis('off')
        fig.canvas.draw()
        
        # make image
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        pil_img = Image.frombytes("RGBA", (w, h), buf, "raw", "ARGB")
        pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        return image
    
    def move_agent(self, pos: tuple, dir: int | None = None):
        self.agent_pos = pos
        self.agent_dir = dir if dir is not None else self.agent_dir
    
    
    def _sanitize(self, pos: tuple):
        return (int(pos[0]), int(pos[1]))
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1
    
    def add_distractors(
        self,
        i: int | None = None,
        j: int | None = None,
        num_distractors: int = 10,
        all_unique: bool = True,
    ):
        # from https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/minigrid_env.py
        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            type = self._rand_elem(["key", "ball"])
            obj = (type, color)

            if all_unique and obj in objs:
                continue


            room_i = i
            room_j = j
            if room_i is None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j is None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists
    
    
        
