import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def init_weights(module, gain=np.sqrt(2)):
    """Khởi tạo trọng số Orthogonal - chuẩn cho RL, giúp tín hiệu lan truyền ổn định."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ActorCritic(nn.Module):
    """Mạng Neural sâu hơn cho cả Actor và Critic."""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        # Phần thân chung - 2 lớp ẩn thay vì 1
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # Đầu Actor: ra xác suất chọn hành động
        self.actor = nn.Linear(hidden_size, action_size)
        # Đầu Critic: ra 1 giá trị đánh giá state
        self.critic = nn.Linear(hidden_size, 1)

        # Áp dụng khởi tạo Orthogonal cho các lớp chung
        self.shared.apply(init_weights)
        # Actor dùng gain nhỏ hơn để xác suất ban đầu gần đều nhau
        init_weights(self.actor, gain=0.01)
        # Critic dùng gain = 1
        init_weights(self.critic, gain=1.0)

    def forward(self, state):
        x = self.shared(state)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class PPO:
    def __init__(self, state_size, action_size, lr=2e-4, gamma=0.99,
                 gae_lambda=0.95, eps_clip=0.2, epochs=4,
                 hidden_size=128, max_grad_norm=0.5,
                 value_loss_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.model = ActorCritic(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Bộ nhớ tạm để lưu trải nghiệm
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        """AI nhìn state và chọn hành động."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Lưu vào bộ nhớ
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_gae(self):
        """
        Tính GAE (Generalized Advantage Estimation).
        GAE giúp cân bằng giữa bias và variance khi ước lượng advantage,
        cho kết quả ổn định hơn nhiều so với tính returns đơn giản.
        Công thức: A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        với delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        advantages = []
        returns = []
        gae = 0

        # Ước lượng giá trị state cuối cùng
        # Nếu episode kết thúc (done), next_value = 0
        # Nếu chưa kết thúc, dùng Critic để ước lượng
        if self.dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(self.states[-1]).unsqueeze(0)
                _, next_value = self.model(state_t)
                next_value = next_value.item()

        # Quét ngược từ cuối về đầu
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Chuẩn hóa advantages (KHÔNG chuẩn hóa returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """Cập nhật trọng số mạng Neural (phần cốt lõi PPO) với GAE."""
        if len(self.states) == 0:
            return 0.0, 0.0

        advantages, returns = self._compute_gae()

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)

        # Train nhiều epoch trên cùng dữ liệu
        for _ in range(self.epochs):
            logits, values = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Tính ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped objective (công thức cốt lõi PPO)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping: ngăn gradient quá lớn gây bất ổn
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Xóa bộ nhớ sau khi học xong
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return policy_loss.item(), value_loss.item()
