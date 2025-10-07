import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = deque(maxlen=50000)
        self.batch_size = 64

    def act(self, state, eps=0.1):
        if random.random() < eps:
            return np.random.randint(self.model.net[-1].out_features)
        else:
            with torch.no_grad():
                return self.model(torch.FloatTensor(state)).argmax().item()

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d)

        q_values = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        next_q = self.target(s2).detach().max(1)[0]
        target_q = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
