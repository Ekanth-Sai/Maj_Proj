import numpy as np
import matplotlib.pyplot as plt

rewards = np.loadtxt("training_rewards.txt")

plt.figure(figsize=(8,5))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve — DQN Robot Reach Task")
plt.grid(True)
plt.tight_layout()
plt.show()
