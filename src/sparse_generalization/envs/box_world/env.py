from copy import deepcopy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import random

from itertools import product
from minigrid.minigrid_env import MiniGridEnv 
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid

from sparse_generalization.envs.box_world.objects import Wall, Goal, Box, Tile, Key
from sparse_generalization.envs.box_world.constants import COLOR_NAMES

# somewhat based on https://github.com/aryangarg794/rnd_dqn_four_room/blob/master/four_room/env.py
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
        highlight=False,
        unsolvable_prob=0.0,
        edge_remove_prob=0.3,
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
    
    def _gen_grid(self, width, height):
        self._current_context = self._list_idx if not self._randomize else None
        self.grid = Grid(width, height)
    
        self.grid.wall_rect(0, 0, width, height)
        
        if self._randomize:
            # waalls: for now simple just one wall
            self.walls = self._simple_wall_gen()
            for wall in self.walls:
                self.grid.set(*wall, Wall())
        
            # agent and goal logic
            self.place_agent()
            self.goal_pos = random.choice(self.valid_pos(margin=1))
            self.grid.set(*self.goal_pos, Goal())
            
            # key-lock logic
            samples = self.num_pairs
            colors = deepcopy(COLOR_NAMES)
            for color in ['green', 'grey', 'red']: # base objs
                colors.remove(color)
            self.edges_one = []
            self.edges_two = []
            pairs_one, colors_one = self._sample_key_pairs(samples, colors)
            for color in colors_one: # no clash with colors
                colors.remove(color)
                
            self.grid.set(*pairs_one[0], Box(colors_one[0]))
            for i, key_lock in enumerate(pairs_one[1:]):
                idx = i+1
                key, lock = key_lock
                self.grid.set(*key, Box(colors_one[idx-1]))
                self.grid.set(*lock, Box(colors_one[idx]))
                self.edges_one.append((i, idx))
                
            
            pairs_two, colors_two = self._sample_key_pairs(samples, colors)
            colors_two[-1] = colors_one[-1]
            
            self.grid.set(*pairs_two[0], Box(colors_two[0]))
            for i, key_lock in enumerate(pairs_two[1:]):
                idx = i+1
                key, lock = key_lock
                self.grid.set(*key, Box(colors_two[idx-1]))
                self.grid.set(*lock, Box(colors_two[idx]))
                self.edges_two.append((i+samples, idx+samples))
                    
            if np.random.rand() < self.unsolvable_prob:
                idx_one = random.choice(list(range(1, self.num_pairs)))
                idx_two = random.choice(list(range(1, self.num_pairs)))
                
                for i in range(1, self.num_pairs):
                    if i == idx_one:
                        key_lock = pairs_one[i]
                        key, lock = key_lock
                        self.grid.set(*key, None)
                        self.grid.set(*lock, None)
                        self.edges_one.remove((i-1, i))
        
                    elif np.random.rand() < self.edge_drop_prob:
                        key_lock = pairs_one[i]
                        key, lock = key_lock
                        self.grid.set(*key, None)
                        self.grid.set(*lock, None)
                        
                        self.edges_one.remove((i-1, i))
                    
                    if i == idx_two:
                        key_lock = pairs_two[i]
                        key, lock = key_lock
                        self.grid.set(*key, None)
                        self.grid.set(*lock, None)
                        self.edges_two.remove((i-1+samples, i+samples))
        
                    elif np.random.rand() < self.edge_drop_prob:
                        key_lock = pairs_two[i]
                        key, lock = key_lock
                        self.grid.set(*key, None)
                        self.grid.set(*lock, None)
                        self.edges_two.remove((i-1+samples, i+samples))
                
            
            self.colors_one = colors_one
            self.colors_two = colors_two
            self.grid.set(self.goal_pos[0]+1, self.goal_pos[1], Box(colors_one[-1]))
                
            
        else: #TODO: from given contexts
            self.walls = self._topology_list[self._list_idx]
            self.agent_pos = self._agent_pos_list[self._list_idx]
            self.agent_dir = self._agent_dir_list[self._list_idx]
            self.keys = self._key_topology_list[self._list_idx]
            self.goal_pos = self._goal_pos_list[self._list_idx]
            
            
        self._current_key = None
        
    def step(self, action):
        self.step_count += 1 
        
    def _simple_wall_gen(self):
        walls = []
        orient = random.randint(0, 1)
        place = random.choice(list(range(2, self.size-2))) # at least 1 space at start, or 2 atr end
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
        valid_pos = self.valid_pos()
        first_key = random.choice(valid_pos)
        pairs.append(first_key)
        valid_pos.remove(first_key)
        state_right = self._sanitize((first_key[0]+1, first_key[1]))
        state_left = self._sanitize((first_key[0]-1, first_key[1]))
        if state_right in valid_pos: valid_pos.remove(state_right)
        if state_left in valid_pos: valid_pos.remove(state_left)
        
        while len(pairs) < num_pairs:
            key = random.choice(valid_pos)
            valid_pos.remove(key)
            if key[0] == 1 or np.array_equal((key[0]-1, key[1]), self.agent_pos): 
                continue
            lock = (key[0]-1, key[1])
            lock_sanitized = self._sanitize(lock)
            if lock_sanitized not in valid_pos: continue
            valid_pos.remove(lock)       
            pairs.append((key, lock))
        
        colors = random.sample(colors_list, k=num_pairs)
            
        return pairs, colors
    
    def valid_pos(self, margin=0):
        valid_pos = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if obj is None:  
                    if margin < x < self.width - margin-1 and margin < y < self.height - margin-1:
                        valid_pos.append((x, y))

        sanitized_pos = self._sanitize(self.agent_pos)
        if sanitized_pos in valid_pos:
            valid_pos.remove(sanitized_pos)
        
        return valid_pos
    
    
    def render_graph(self, figsize=(10, 6)):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        img = self.render()
        ax[0].imshow(img)
        
        def custom_layout(G, num_pairs, y1=0.0, y2=-0.5, y_goal=1.0, spacing=0.5):
            pos = {}
            for i in range(num_pairs):
                pos[i] = (i * spacing, y1)
            for i in range(num_pairs):
                pos[num_pairs + i] = (i * spacing, y2)
            center_x = ((num_pairs) * spacing)
            y_mid = (y1 + y2) / 2
            pos[num_pairs * 2] = (center_x, y_mid)
            return pos
        
        # draw graph 
        edges = self.edges_one + self.edges_two
        colors = self.colors_one + self.colors_two + ['springgreen']
        nodes = [i for i in range(len(colors))] 
        nodes = nodes + [self.num_pairs*2]
        edges = edges + [(self.num_pairs-1, self.num_pairs*2), (self.num_pairs*2-1, self.num_pairs*2)]
        
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        pos = custom_layout(G, self.num_pairs)
        nx.draw_networkx_nodes(G, pos, node_color=colors, ax=ax[1], node_size=1000, linewidths=1, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax[1])
        
        plt.tight_layout()
        plt.axis('off')
        
        return
    
    
    def _sanitize(self, pos):
        return (int(pos[0]), int(pos[1]))
    
    
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
    
    def reset(self, *, seed=None, options=None):
        if seed and not self._randomize:
            random.seed(seed)
            self._list_idx = random.randint(0, self._list_size-1)
        else:
            pass
            
        return super().reset(seed=seed, options=options)
        
