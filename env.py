from __future__ import annotations
import gymnasium as gym
from gym.utils import seeding
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box
from minigrid.minigrid_env import MiniGridEnv
from enum import Enum
import pygame
import numpy as np


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

    def __init__(
            self,
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            max_steps: int | None = None,
            max_reward_episodes: int = 20,  # Number of episodes with sword reward
            **kwargs,
        ):
            self.step_count = 0
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir

            self.max_reward_episodes = max_reward_episodes  # Threshold for giving sword reward
            self.current_episode = 0  # Track the current episode

            # Track which resources have been collected during the entire training
            self.collected_resources_global = set()

            # Tracking if the sword has been crafted this episode
            self.sword_crafted = False

            self.resource_names = ["Iron Ore", "Silver Ore", "Platinum Ore", "Gold Ore", "Tree", "Chest", "Crafting Table", "Wall"]

            self.inventory = []
            mission_space = MissionSpace(mission_func=self._gen_mission)

            if max_steps is None:
                max_steps = 4 * size**2

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
                **kwargs,
            )

            self.action_space = gym.spaces.Discrete(len(self.Actions))

            lidar_shape = (8, len(self.resource_names))  # 8 beams, each detecting one of the 8 possible entities
            self.observation_space = gym.spaces.Dict({
                "lidar": gym.spaces.Box(low=0, high=1, shape=lidar_shape, dtype=np.float32),
                "inventory": gym.spaces.MultiDiscrete([10]*len(self.resource_names))  # Maximum 10 of each item in inventory
            })


    @staticmethod
    def _gen_mission():
        return "Collect resources, craft a sword, and find the treasure."

    def _gen_grid(self, width, height):
        # Create an empty grid and build the walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place resources and other objects on the grid
        self.place_obj(Resource("red", "Iron Ore"), top=(1, 1))
        self.place_obj(Resource("grey", "Silver Ore"), top=(2, 1))
        self.place_obj(Resource("purple", "Platinum Ore"), top=(3, 1))
        self.place_obj(Resource("yellow", "Gold Ore"), top=(4, 1))
        self.place_obj(Resource("green", "Tree"), top=(5, 1))
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
        lidar_obs = np.zeros((8, len(self.resource_names)))  # 8 beams, each with [object_type, distance]
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        for i, angle in enumerate(angles):
            min_dist = float('inf')
            closest_entity_idx = -1

            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    obj = self.grid.get(x, y)
                    if obj is not None:
                        obj_pos = np.array([x, y])
                        agent_pos = np.array(self.agent_pos)
                        vec_to_obj = obj_pos - agent_pos
                        dist_to_obj = np.linalg.norm(vec_to_obj)
                        angle_to_obj = np.arctan2(vec_to_obj[1], vec_to_obj[0])

                        angle_diff = (angle_to_obj - angle + np.pi) % (2 * np.pi) - np.pi

                        if abs(angle_diff) <= np.pi / 8 and dist_to_obj < min_dist:
                            min_dist = dist_to_obj
                            closest_entity_idx = self.get_entity_index(obj)

            if closest_entity_idx != -1:
                lidar_obs[i, closest_entity_idx] = min_dist / self.grid.width  # Normalize distance by grid width

        return lidar_obs

    def get_entity_index(self, obj):
        # Map object to the corresponding index in self.resource_names
        if isinstance(obj, Resource):
            return self.resource_names.index(obj.resource_name)
        elif isinstance(obj, Box):
            return self.resource_names.index("Chest" if obj.color == 'purple' else "Crafting Table")
        else:
            return self.resource_names.index("Wall")

    def get_inventory_observation(self):
        inventory_obs = np.zeros(len(self.resource_names), dtype=np.float32)
        for item in self.inventory:
            if item in self.resource_names:
                index = self.resource_names.index(item)
                inventory_obs[index] += 1
        return inventory_obs

    def get_obs(self):
        lidar_obs = self.get_lidar_observation().flatten().astype(np.float32)   # Flatten lidar
        inventory_obs = self.get_inventory_observation().astype(np.float32)

        # print (f"inventory ={inventory_obs}")
        # print(f"lidar_obs ={lidar_obs}")

        # Concatenate lidar and inventory observations
        combined_obs = np.concatenate((lidar_obs, inventory_obs), axis=0).astype(np.float32)
        # print(f"combined_obs ={combined_obs.shape}")

        return combined_obs

    # def step(self, action):
        # Custom action for crafting the sword
        if action == self.Actions.craft_sword.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            # Check if the agent is in front of the crafting table and has required items
            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table
                if "Tree" in self.inventory and "Iron Ore" in self.inventory:
                    # Remove the required resources
                    self.inventory.remove("Tree")
                    self.inventory.remove("Iron Ore")
                    # Add the crafted item to the inventory
                    self.inventory.append("Iron Sword")
                    print("Crafted an Iron Sword!")
                    reward = 10  # Reward for crafting
                    terminated = False
                    truncated = False
                    return self.get_obs(), reward, terminated, truncated, {}

                else:
                    # print("You need Tree and Iron Ore to craft a sword.")
                    return self.get_obs(), -1, False, False, {}

        # Custom action for opening the chest
        elif action == self.Actions.open_chest.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            # Check if the agent is in front of the chest and has an Iron Sword
            if isinstance(fwd_cell, Box) and fwd_cell.color == 'purple':  # Chest
                if "Iron Sword" in self.inventory:
                    # Add the treasure to the inventory and end the game
                    self.inventory.append("Treasure")
                    print("Found the treasure! You win!")
                    reward = 1000  # Large reward for finding the treasure
                    terminated = True  # End the game
                    truncated = False
                    return self.get_obs(), reward, terminated, truncated, {}

                else:
                    # print("You need an Iron Sword to open the chest.")
                    return self.get_obs(), -1, False, False, {}

        # Handle the toggle action for collecting resources or interacting with boxes
        elif action == self.Actions.toggle.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            # Only collect Resource objects, not Box objects
            if isinstance(fwd_cell, Resource):
                self.inventory.append(fwd_cell.resource_name)
                self.grid.set(*fwd_pos, None)  # Remove the object from the grid
                reward = 1
                terminated = False
                truncated = False
                return self.get_obs(), reward, terminated, truncated, {}

            # If it's a Box (like chest or crafting table), don't allow it to be collected
            elif isinstance(fwd_cell, Box):
                # print(f"Interacted with {fwd_cell.color} box but cannot collect.")
                return self.get_obs(), -1, False, False, {}

        # Fallback to the parent class's step function for basic actions (move, turn, etc.)
        self.step_count += 1  # Keep track of step count
        obs, reward, terminated, truncated, info = super().step(action)

        # Override the observation to return the custom observation format (lidar + inventory)
        # print (reward, terminated, truncated, info)
        return self.get_obs(), reward, terminated, truncated, info

    def step(self, action):
        reward = -1  # Default time step penalty

        # Custom action for crafting the sword
        if action == self.Actions.craft_sword.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table
                if "Tree" in self.inventory and "Iron Ore" in self.inventory and not self.sword_crafted:
                    self.inventory.remove("Tree")
                    self.inventory.remove("Iron Ore")
                    self.inventory.append("Iron Sword")
                    print("Crafted an Iron Sword!")
                    self.sword_crafted = True
                    if self.current_episode < self.max_reward_episodes:
                        reward += 50  # Reward for crafting the sword within the first `max_reward_episodes` episodes
                else:
                    reward += -1  # Penalize failed crafting attempt
                return self.get_obs(), reward, False, False, {}

        # Custom action for opening the chest
        elif action == self.Actions.open_chest.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'purple':  # Chest
                if "Iron Sword" in self.inventory:
                    self.inventory.append("Treasure")
                    print("Found the treasure! You win!")
                    reward += 1000  # Large reward for reaching the goal
                    return self.get_obs(), reward, True, False, {}  # End the game
                else:
                    reward += -1  # Penalize for not having the sword
                return self.get_obs(), reward, False, False, {}

        # Handle the toggle action for collecting resources
        elif action == self.Actions.toggle.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Resource):
                if fwd_cell.resource_name not in self.collected_resources_global:
                    self.collected_resources_global.add(fwd_cell.resource_name)
                    self.inventory.append(fwd_cell.resource_name)
                    self.grid.set(*fwd_pos, None)
                    reward += 1  # Reward for collecting the resource for the first time during training
                else:
                    reward += -1  # Penalize for redundant resource collection
                return self.get_obs(), reward, False, False, {}

        # Per step penalty
        obs, reward_super, terminated, truncated, info = super().step(action)
        reward += reward_super

        return self.get_obs(), reward, terminated, truncated, info    

    def reset(self, seed=None, **kwargs):
        self.np_random, seed = seeding.np_random(seed)
        self.inventory = []
        self.sword_crafted = False  # Reset sword crafting per episode

        # Increase the episode count
        self.current_episode += 1

        self._gen_grid(self.width, self.height)
        self.place_agent()
        self.step_count = 0
        return self.get_obs(), {}

        # Return the custom observation format
        return self.get_obs(), {}

    def render(self):
        # Call the parent class's render method
        result = super().render()

        # Display the agent's inventory on the screen (in the terminal or add GUI)
        print(f"Inventory: {', '.join(self.inventory)}")

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
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

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