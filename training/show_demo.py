import torch
import numpy as np
import time
from envs.mujoco_reach_env import MujocoReachEnv
from agents.dqn_agent import DQNAgent

# Load environment and trained model
env = MujocoReachEnv()
agent = DQNAgent(state_dim=4, action_dim=5)
agent.model.load_state_dict(torch.load("trained_reach_agent.pth"))
agent.model.eval()

print("\n🎬 Running trained DQN agent demo...\n")

test_episodes = 3
for ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n▶️ Episode {ep+1} | Target: {state[2:]}")
    while not done:
        action = agent.act(state, eps=0.0)  # no randomness
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        step += 1
        env.render()
        time.sleep(0.05)

    print(f"🏁 Episode {ep+1} finished in {step} steps | Total Reward: {total_reward:.2f}")

print("\n✅ Demo complete!")
