"""
DQN Agent for Multi-Agent traffic light control (Parameter Sharing).
So sánh baseline với PPO.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent với Parameter Sharing cho multi-agent.
    
    Một Q-network dùng chung cho tất cả agents.
    Hỗ trợ Double DQN + Prioritized Replay thông qua loss weighting.
    """

    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_size=50000, batch_size=64, target_update_freq=100, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.steps = 0

        # Q-network + Target network (cho Double DQN)
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

        # Metrics tracking
        self.losses = []

    def get_q_values(self, state, can_switch=True):
        """Tính Q-values với action masking."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)

        if not can_switch:
            q_values[0, 1] = -1e9  # Mask action 1 (switch)

        return q_values.squeeze(0).numpy()

    def select_action(self, state, can_switch=True):
        """Epsilon-greedy với action masking."""
        if np.random.random() < self.epsilon:
            if not can_switch:
                return 0
            return np.random.randint(0, self.action_size)

        q = self.get_q_values(state, can_switch)
        return int(q.argmax())

    def store_experience(self, state, action, reward, next_state, done):
        """Lưu experience vào replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """Mini-batch training từ replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q (Double DQN): Q_target(s', argmax Q_online(s', a'))
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            for tp, qp in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.copy_(self.tau * qp.data + (1.0 - self.tau) * tp.data)

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, weights_only=True))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()
        self.target_network.eval()
