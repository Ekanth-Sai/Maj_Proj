import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from envs.mujoco_reach_env import MujocoReachEnv
from agents.dqn_agent import DQNAgent

# ===============================
# 🧠 Load trained model
# ===============================
env = MujocoReachEnv()
agent = DQNAgent(state_dim=4, action_dim=5)
agent.model.load_state_dict(torch.load("trained_reach_agent.pth"))
agent.model.eval()

# ===============================
# 🎨 Initialize Matplotlib (TkAgg)
# ===============================
plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("RL Robot Reaching Task (DQN Live Demo)")
arm_dot, = ax.plot([], [], 'bo', label='Arm', markersize=10)
target_dot, = ax.plot([], [], 'ro', label='Target', markersize=10)
trail_line, = ax.plot([], [], 'b--', alpha=0.4, label='Trail')
plt.legend()

# 🎞️ Store frames for GIF
frames = []

# ===============================
# 🏃 Run Test Episodes
# ===============================
test_episodes = 2
for ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    arm_pos = state[:2]
    target_pos = state[2:]
    trail_x, trail_y = [], []

    print(f"\n▶️ Episode {ep+1} | Target: {target_pos}")

    while not done:
        action = agent.act(state, eps=0.0)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        step += 1

        arm_pos = state[:2]
        target_pos = state[2:]

        trail_x.append(arm_pos[0])
        trail_y.append(arm_pos[1])

        arm_dot.set_data([arm_pos[0]], [arm_pos[1]])
        target_dot.set_data([target_pos[0]], [target_pos[1]])
        trail_line.set_data(trail_x, trail_y)
        plt.pause(0.03)

        # ✅ Convert ARGB → RGB safely (works with TkAgg)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        rgb_buf = buf[:, :, [1, 2, 3]]  # reorder channels ARGB → RGB
        frames.append(rgb_buf)

    print(f"🏁 Episode {ep+1} finished in {step} steps | Total Reward = {total_reward:.2f}")

plt.ioff()
plt.show()

# ===============================
# 💾 Save Animation as GIF
# ===============================
output_path = "robot_reach_demo.gif"
imageio.mimsave(output_path, frames, fps=20)
print(f"\n🎞️ Saved animation as {output_path}")
