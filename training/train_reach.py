import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from envs.mujoco_reach_env import MujocoReachEnv
from agents.dqn_agent import DQNAgent
import numpy as np
from tqdm import trange

env = MujocoReachEnv()
agent = DQNAgent(state_dim=4, action_dim=5)

episodes = 200
for ep in trange(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    if ep % 20 == 0:
        print(f"Episode {ep}: Total reward {total_reward:.2f}")
