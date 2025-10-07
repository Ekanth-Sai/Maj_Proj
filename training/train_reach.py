import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.mujoco_reach_env import MujocoReachEnv
from agents.dqn_agent import DQNAgent
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch

# Initialize environment and agent
env = MujocoReachEnv()
agent = DQNAgent(state_dim=4, action_dim=5)

episodes = 200
all_rewards = []  # 🧠 Track total reward per episode

# ===============================
# 🏋️ TRAINING LOOP
# ===============================
for ep in trange(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Agent selects an action (ε-greedy)
        action = agent.act(state)

        # Environment step
        next_state, reward, done, _, _ = env.step(action)

        # Store experience and train
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        # Update state and cumulative reward
        state = next_state
        total_reward += reward

    # Store total reward
    all_rewards.append(total_reward)

    # Print progress occasionally
    if ep % 20 == 0:
        print(f"Episode {ep}: Total reward {total_reward:.2f}")

# ===============================
# 📈 POST-TRAINING STATS
# ===============================
print("\n✅ Training complete!")
print(f"Best episode reward: {max(all_rewards):.2f}")
print(f"Worst episode reward: {min(all_rewards):.2f}")
print(f"Average reward (last 20 episodes): {np.mean(all_rewards[-20:]):.2f}")

# ===============================
# 📊 PLOT LEARNING CURVE
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(all_rewards, label='Total Reward per Episode')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve — RL Robot Reaching Task")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 💾 SAVE TRAINED MODEL (Optimized)
# ===============================

# Move model to CPU before saving (prevents GPU->CPU lag on hybrid systems)
agent.model.to("cpu")

# # Free replay buffer memory (optional, reduces memory pressure during save)
agent.memory.clear()

# Save rewards log for later plotting or report
np.savetxt("training_rewards.txt", all_rewards)
print("\n📄 Saved training rewards to 'training_rewards.txt'")

# Save model state dictionary
print("💾 Saving trained model... (this may take a few seconds)")
torch.save(agent.model.state_dict(), "trained_reach_agent.pth")
print("✅ Model saved successfully as 'trained_reach_agent.pth'")

# ===============================
# 🎯 TEST TRAINED AGENT (Improved Output)
# ===============================
print("\n🎯 Evaluating trained agent (no exploration)...\n")

test_episodes = 5
test_rewards = []

for ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n▶️ Test Episode {ep+1}")
    print(f"Initial Arm: {state[:2]}, Target: {state[2:]}")

    while not done:
        action = agent.act(state, eps=0.0)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        step += 1
        if step % 5 == 0:  # print every 5 steps
            env.render()

    test_rewards.append(total_reward)
    print(f"🏁 Episode {ep+1} finished in {step} steps, Total Reward = {total_reward:.2f}")

print(f"\n🌟 Average Test Reward: {np.mean(test_rewards):.2f}")
