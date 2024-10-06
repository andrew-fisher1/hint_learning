from __future__ import annotations
import gymnasium as gym
from gymnasium.utils import seeding
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from enum import Enum
import pygame
import numpy as np


class RewardType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"

class Resource(Ball):
    """Custom Ball object with a resource name"""
    def __init__(self, color, resource_name):
        super().__init__(color)
        self.resource_name = resource_name

class SimpleEnv(MiniGridEnv):
    class Actions(Enum):
        move_forward = 0
        turn_left = 1
        turn_right = 2
        toggle = 3
        craft_sword = 4
        open_chest = 5
        approach_crafting_table = 6
        approach_chest = 7
        approach_tree = 8
        approach_iron_ore = 9
        approach_tree = 8
        approach_iron_ore = 9

    def __init__(
            self,
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            reward_type: RewardType = RewardType.SPARSE,
            max_steps: int | None = None,
            max_reward_episodes: int = 50,  # Number of episodes with sword reward
            **kwargs,
        ):
            self.step_count = 0
            self.cumulative_reward = 0
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.reward_type = reward_type

            self.max_reward_episodes = max_reward_episodes  # Threshold for giving sword reward
            self.current_episode = 0  # Track the current episode

            self.sword_crafted = False

            # Track which resources have been collected during the entire training
            self.collected_resources_global = set()

            # Updated the resource_names to reflect only non-collected world items
            self.resource_names = ["iron_ore", "silver_ore", "platinum_ore", "gold_ore", "tree", "chest", "crafting_table", "wall"]

            # Inventory for collected items
            self.inventory_items = ["iron", "silver", "gold", "platinum", "wood", "iron_sword", "treasure"]

            self.inventory = []
            mission_space = MissionSpace(mission_func=self._gen_mission)

            self.crafted_sword_episodes = 0
            self.collected_resource_episodes = {
                "tree": 0,
                "iron_ore": 0,
                "silver_ore": 0,
                "gold_ore": 0,
                "platinum_ore": 0
            }
            # if max_steps is None:
            #     max_steps = 4 * size**2
            if max_steps is None:
                max_steps = 300

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
                **kwargs,
            )

            self.action_space = gym.spaces.Discrete(len(self.Actions))
            # print (f"Actions space in the constructor = {self.action_space}")

                    # Initialize lidar and inventory observation shapes
            lidar_shape = (8, len(self.resource_names))  # 8 lidar beams, each detecting one of the entities
            inventory_shape = (len(self.inventory_items),)

            # Set up observation space based on lidar and inventory
            self.observation_space = gym.spaces.Dict({
                "lidar": gym.spaces.Box(low=0, high=1, shape=lidar_shape, dtype=np.float32),
                "inventory": gym.spaces.MultiDiscrete([10] * len(self.inventory_items))  # Max 10 of each item
            })

            # print(f"Observation space set up: {self.observation_space}")

            # Adjusted lidar observation to focus on the non-collected world items
            lidar_shape = (8, len(self.resource_names))  # 8 beams, each detecting one of the 8 possible entities
            self.observation_space = gym.spaces.Dict({
                "lidar": gym.spaces.Box(low=0, high=1, shape=lidar_shape, dtype=np.float32),
                "inventory": gym.spaces.MultiDiscrete([10] * len(self.inventory_items))  # Maximum 10 of each item in inventory

            })



    @staticmethod
    def _gen_mission():
        return "Collect resources, craft a sword, and find the treasure."

    def _gen_grid(self, width, height):
        # Create an empty grid and build the walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place resources and other objects on the grid
        self.place_obj(Resource("red", "iron_ore"), top=(1, 1))
        self.place_obj(Resource("grey", "silver_ore"), top=(2, 1))
        self.place_obj(Resource("purple", "platinum_ore"), top=(3, 1))
        self.place_obj(Resource("yellow", "gold_ore"), top=(4, 1))
        self.place_obj(Resource("green", "tree"), top=(5, 1))
        self.place_obj(Box("purple"), top=(6, 1))  # Chest
        self.place_obj(Box("blue"), top=(7, 1))    # Crafting table

        # Ensure the agent's starting position is placed in a valid, empty cell
        if self.agent_start_pos is not None:
            start_cell = self.grid.get(*self.agent_start_pos)
            if start_cell is not None and not start_cell.can_overlap():
                # Find a new valid position for the agent if the start position is occupied
                empty_positions = [
                    (x, y) for x in range(1, width - 1)
                    for y in range(1, height - 1)
                    if self.grid.get(x, y) is None
                ]
                if empty_positions:
                    self.agent_start_pos = empty_positions[0]  # Set to the first empty position
                else:
                    raise RuntimeError("No valid starting position available for the agent.")
            
            # Place the agent in the valid position
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()  # Place the agent randomly if no start position is specified

    def get_lidar_observation(self):
        # Initialize the lidar observation matrix: 8 beams x number of object types
        lidar_obs = np.zeros((8, len(self.resource_names)), dtype=np.float32)
        
        # Define the angles for each beam
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        # Loop through each beam
        for i, angle in enumerate(angles):
            x, y = self.agent_pos  # Start position of the agent
            x_ratio, y_ratio = np.cos(angle), np.sin(angle)

            # Continue shooting the beam until it hits a wall or exits the grid
            for beam_range in range(1, self.grid.width + 1):  # Maximum range based on the grid size
                x_obj = int(x + beam_range * x_ratio)
                y_obj = int(y + beam_range * y_ratio)

                # Check if the beam is out of bounds
                if not (0 <= x_obj < self.grid.width and 0 <= y_obj < self.grid.height):
                    break  # Beam has exited the grid

                # Get the object at the current beam position
                obj = self.grid.get(x_obj, y_obj)

                if obj is not None:
                    # Get the index of the detected object
                    closest_entity_idx = self.get_entity_index(obj)
                    
                    # Record the distance in the corresponding beam and object type channel
                    if lidar_obs[i, closest_entity_idx] == 0:  # Only update if no previous object was detected on this beam
                        lidar_obs[i, closest_entity_idx] = beam_range / self.grid.width  # Normalize the distance

                # Stop the beam if the object is a wall (can_occlude)
                    if obj.type == "wall":  # Checking if the object is a wall by its type
                        break  # Stop the beam as it hit a wall

        return lidar_obs

    def get_entity_index(self, obj):
        # Map object to the corresponding index in self.resource_names
        if isinstance(obj, Resource):
            return self.resource_names.index(obj.resource_name)
        elif isinstance(obj, Box):
            return self.resource_names.index("chest" if obj.color == 'purple' else "crafting_table")
        else:
            return self.resource_names.index("wall")

    def get_inventory_observation(self):
        # Updated to only track items that can be added to inventory
        inventory_obs = np.zeros(len(self.inventory_items), dtype=np.float32)
        # print (f"Inventory  = {self.inventory}")
        # print (f"inventory_items = {self.inventory_items}")
        for item in self.inventory:
            if item in self.inventory_items:
                index = self.inventory_items.index(item)
                inventory_obs[index] += 1
        return inventory_obs

    def get_obs(self):
        lidar_obs = self.get_lidar_observation().flatten().astype(np.float32)   # Flatten lidar
        inventory_obs = self.get_inventory_observation().astype(np.float32)

        # Debugging prints for lidar and inventory shapes
        # print(f"Debug: Lidar Observation Shape: {lidar_obs.shape}")
        # print(f"Debug: Inventory Observation Shape: {inventory_obs.shape}")

        # Concatenate lidar and inventory observations
        combined_obs = np.concatenate((lidar_obs, inventory_obs), axis=0).astype(np.float32)
        # print(f"Debug: Combined Observation Shape: {combined_obs.shape}")

        return combined_obs
    
    # utility function to find path to object
    def find_object_position(self, obj_name):
        """Find the position of the object in the grid by its name."""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Resource) and obj.resource_name == obj_name:
                    return (x, y)
                elif isinstance(obj, Box) and obj_name == "chest" and obj.color == "purple":
                    return (x, y)
                elif isinstance(obj, Box) and obj_name == "crafting_table" and obj.color == "blue":
                    return (x, y)
        return None
    def get_adjacent_pos_and_dir(self, agent_pos, target_pos):
        """Given the agent's position and the target position, find the adjacent position and direction to face the object."""
        x, y = target_pos

        # List of adjacent positions and the direction the agent should face to look at the target object
        adjacent_positions = [
            ((x - 1, y), 0),  # Left of the object, face right
            ((x + 1, y), 2),  # Right of the object, face left
            ((x, y - 1), 3),  # Below the object, face up
            ((x, y + 1), 1),  # Above the object, face down
        ]

        # Loop through the possible adjacent positions
        for adj_pos, adj_dir in adjacent_positions:
            # Ensure the adjacent cell is valid (empty or movable) and within bounds
            if 0 <= adj_pos[0] < self.grid.width and 0 <= adj_pos[1] < self.grid.height:
                adj_cell = self.grid.get(*adj_pos)
                if adj_cell is None:  # Ensure the adjacent cell is empty
                    return adj_pos, adj_dir

        return None, None  # Return None if no valid adjacent position is found
    
    def step(self, action):
        reward = -0.1  # Default time step penalty
        self.cumulative_reward += reward  # Track the cumulative reward
        terminated = False
        truncated = False

        # Custom action for approaching tree
        if action == self.Actions.approach_tree.value:
            # print("Executing approach_tree action")
            crafting_table_pos = self.find_object_position("tree")
            if crafting_table_pos:
                adj_pos, adj_dir = self.get_adjacent_pos_and_dir(self.agent_pos, crafting_table_pos)
                if adj_pos is not None:
                    self.agent_pos = adj_pos  # Teleport the agent to the adjacent position
                    self.agent_dir = adj_dir  # Make the agent face the object
            return self.get_obs(), reward, terminated, truncated, {}

        # Custom action for approaching iron_ore
        if action == self.Actions.approach_iron_ore.value:
            # print("Executing approach_iron_ore action")
            crafting_table_pos = self.find_object_position("iron_ore")
            if crafting_table_pos:
                adj_pos, adj_dir = self.get_adjacent_pos_and_dir(self.agent_pos, crafting_table_pos)
                if adj_pos is not None:
                    self.agent_pos = adj_pos  # Teleport the agent to the adjacent position
                    self.agent_dir = adj_dir  # Make the agent face the object
            return self.get_obs(), reward, terminated, truncated, {}

        # Custom action for approaching crafting table
        if action == self.Actions.approach_crafting_table.value:
            # print("Executing approach_crafting_table action")
            crafting_table_pos = self.find_object_position("crafting_table")
            if crafting_table_pos:
                adj_pos, adj_dir = self.get_adjacent_pos_and_dir(self.agent_pos, crafting_table_pos)
                if adj_pos is not None:
                    self.agent_pos = adj_pos  # Teleport the agent to the adjacent position
                    self.agent_dir = adj_dir  # Make the agent face the object
            return self.get_obs(), reward, terminated, truncated, {}

        # Custom action for approaching chest
        elif action == self.Actions.approach_chest.value:
            # print("Executing approach_chest action")
            chest_pos = self.find_object_position("chest")
            if chest_pos:
                adj_pos, adj_dir = self.get_adjacent_pos_and_dir(self.agent_pos, chest_pos)
                if adj_pos is not None:
                    self.agent_pos = adj_pos  # Teleport the agent to the adjacent position
                    self.agent_dir = adj_dir  # Make the agent face the object
            return self.get_obs(), reward, terminated, truncated, {}

        # Action for crafting the sword
        elif action == self.Actions.craft_sword.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table
                if "wood" in self.inventory and "iron" in self.inventory and not self.sword_crafted:
                    self.inventory.remove("wood")
                    self.inventory.remove("iron")
                    self.inventory.append("iron_sword")
                    # print("Crafted an Iron Sword!")
                    self.sword_crafted = True
                    self.crafted_sword_episodes += 1
                    if self.reward_type == RewardType.DENSE:
                        if self.crafted_sword_episodes < self.max_reward_episodes:
                            reward = 50  # Reward for crafting the sword
                        else:
                            reward = 5
                    else:  # SPARSE reward: No reward until the final goal state
                        reward = -0.1
            return self.get_obs(), reward, terminated, truncated, {}

        # Action for opening the chest
        elif action == self.Actions.open_chest.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'purple':  # Chest
                if "iron_sword" in self.inventory:
                    self.inventory.append("treasure")
                    # print("Found the treasure! You win!")
                    # print("Reached Goal State")
                    if self.reward_type == RewardType.SPARSE:
                        reward = 1000  # Large reward for finding the treasure (sparse reward)
                    else:
                        reward = 1000  # Large reward for finding the treasure (dense reward)
                    terminated = True
            return self.get_obs(), reward, terminated, truncated, {}

        elif action == self.Actions.toggle.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is None:
                return self.get_obs(), reward, terminated, truncated, {}

            # Only allow Resource objects to be collected
            if isinstance(fwd_cell, Resource):
                resource_name = fwd_cell.resource_name
                if resource_name not in self.collected_resources_global:
                    self.collected_resources_global.add(resource_name)
                    # Map the resource to its inventory name
                    if resource_name == "iron_ore":
                        self.inventory.append("iron")
                    elif resource_name == "silver_ore":
                        self.inventory.append("silver")
                    elif resource_name == "gold_ore":
                        self.inventory.append("gold")
                    elif resource_name == "tree":
                        self.inventory.append("wood")

                    # Update the lidar observation and inventory
                    self.grid.set(*fwd_pos, None)  # Remove the object from the grid
                    # reward = 1  # Reward for collecting the resource
                    if self.reward_type == RewardType.SPARSE:
                        reward = -0.1
                    else:
                        if self.collected_resource_episodes[resource_name] < self.max_reward_episodes:
                            reward = 10  # Reward for crafting the sword
                        else:
                            reward = 1
                    self.cumulative_reward += reward
                else:
                    reward = -0.1  # Penalize redundant collection within the same episode
            return self.get_obs(), reward, terminated, truncated, {}

        # Handle basic actions (move, turn, etc.) using the parent class
        if action in [self.Actions.move_forward.value, self.Actions.turn_left.value, self.Actions.turn_right.value]:
            self.step_count += 1  # Keep track of step count
            obs, reward_super, terminated, truncated, info = super().step(action)
            reward += reward_super
            self.cumulative_reward += reward  # Update cumulative reward
            return self.get_obs(), reward, terminated, truncated, info

        # Handle unknown actions
        else:
            raise ValueError(f"Unknown action: {action}")
   
    def reset(self, seed=None, **kwargs):
        # print (f"Cumulative reward: {self.cumulative_reward}")
        # print (f"episodes for crafting sword = {self.crafted_sword_episodes}")
        self.np_random, seed = seeding.np_random(seed)
        self.inventory = []
        self.sword_crafted = False  # Reset sword crafting per episode

        # Reset per-episode resource collection
        self.collected_resources_global = set()

        # Increase the episode count
        self.current_episode += 1
        self.cumulative_reward = 0

        self._gen_grid(self.width, self.height)
        self.place_agent()
        self.step_count = 0
        return self.get_obs(), {}

    def render(self):
        # Call the parent class's render method
        result = super().render()

        # Display the agent's inventory on the screen (in the terminal or add GUI)
        # print(f"Inventory: {', '.join(self.inventory)}")

        return result


# Custom manual control class for handling custom actions
class CustomManualControl:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print (f"Action = {action})")
        print (f"obs = {obs}")
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        print (f"self.cumulated reward = {self.env.cumulative_reward:.2f}")


        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()


    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": SimpleEnv.Actions.turn_left.value,
            "right": SimpleEnv.Actions.turn_right.value,
            "up": SimpleEnv.Actions.move_forward.value,
            "space": SimpleEnv.Actions.toggle.value,
            "c": SimpleEnv.Actions.craft_sword.value,  # 'c' for craft sword
            "o": SimpleEnv.Actions.open_chest.value,   # 'o' for open chest
            "t": SimpleEnv.Actions.approach_crafting_table.value,  # 't' for approach crafting table
            "h": SimpleEnv.Actions.approach_chest.value,  # 'h' for approach chest        
            }

        if key in key_to_action:
            action = key_to_action[key]
            self.step(action)
        else:
            print("Unmapped key:", key)


def main():
    env = SimpleEnv(render_mode="human")
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()