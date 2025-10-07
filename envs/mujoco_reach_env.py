import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MujocoReachEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Discrete(5)  # Corresponds to 5 discrete actions
        self.target = np.array([0.5, 0.5])
        self.arm_state = np.zeros(2)
        self.action_map = {
            0: np.array([0.1, 0]),    # Move right
            1: np.array([-0.1, 0]),   # Move left
            2: np.array([0, 0.1]),    # Move up
            3: np.array([0, -0.1]),   # Move down
            4: np.array([0, 0])       # Stay
        }

    def step(self, action):
        continuous_action = self.action_map[action]
        self.arm_state = np.clip(self.arm_state + continuous_action * 0.05, -1, 1)
        dist = np.linalg.norm(self.arm_state - self.target)
        reward = -10 * dist   # scaled for stronger signal

        done = dist < 0.02    # tighter success condition
        if done:
            reward += 10.0    # success bonus

        return np.concatenate([self.arm_state, self.target]), reward, done, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arm_state = np.random.uniform(-1, 1, size=2)
        self.target = np.random.uniform(-1, 1, size=2)
        return np.concatenate([self.arm_state, self.target]), {}

    def render(self):
        print(f"Arm: {self.arm_state}, Target: {self.target}")
