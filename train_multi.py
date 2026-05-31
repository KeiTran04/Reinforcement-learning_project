import matplotlib.pyplot as plt
from multi_sumo_env import MultiSumoEnv
from ppo_agent import PPO, ActorCritic
import numpy as np
import torch
import torch.nn as nn
import os

class MultiAgentPPO(PPO):
    """PPO mở rộng cho Multi-Agent với Parameter Sharing + Schedule."""
    
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, eps_clip=0.2, epochs=6,
                 hidden_size=128, max_grad_norm=0.5,
                 value_loss_coef=0.5, entropy_coef=0.03):
        super().__init__(state_size, action_size, lr, gamma, gae_lambda,
                         eps_clip, epochs, hidden_size, max_grad_norm,
                         value_loss_coef, entropy_coef)
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.dones.clear(); self.values.clear()
        
        self.agent_buffers = {}
        self.initial_lr = lr
        self.initial_entropy_coef = entropy_coef
        
    def init_agent_buffer(self, tl_id):
        self.agent_buffers[tl_id] = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'dones': [], 'values': []
        }
        
    def update_schedule(self, progress):
        """LR và entropy giảm dần theo tiến trình (0.0 → 1.0)."""
        lr = self.initial_lr * (1.0 - 0.9 * progress)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.entropy_coef = self.initial_entropy_coef * (1.0 - 0.5 * progress)
        
    def select_action_for_agent(self, tl_id, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)
            
        # Action Masking: Nếu không được phép đổi (cờ can_switch ở index 6 = 0.0), set logit của Action 1 = -1e9
        can_switch = state[6] > 0.5
        if not can_switch:
            logits = logits.clone()
            logits[0, 1] = -1e9
            
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if tl_id not in self.agent_buffers:
            self.init_agent_buffer(tl_id)
        buf = self.agent_buffers[tl_id]
        buf['states'].append(state)
        buf['actions'].append(action.item())
        buf['log_probs'].append(log_prob.item())
        buf['values'].append(value.item())
        return action.item()
        
    def store_reward_for_agent(self, tl_id, reward, done):
        if tl_id not in self.agent_buffers:
            self.init_agent_buffer(tl_id)
        self.agent_buffers[tl_id]['rewards'].append(reward)
        self.agent_buffers[tl_id]['dones'].append(done)
        
    def _compute_gae_for_agent(self, tl_id):
        buf = self.agent_buffers[tl_id]
        if len(buf['states']) == 0:
            return [], []
            
        next_value = 0
        if not buf['dones'][-1]:
            with torch.no_grad():
                _, nv = self.model(torch.FloatTensor(buf['states'][-1]).unsqueeze(0))
                next_value = nv.item()
                
        values = buf['values'] + [next_value]
        advantages, returns = [], []
        gae = 0
        
        for t in reversed(range(len(buf['rewards']))):
            if buf['dones'][t]:
                delta = buf['rewards'][t] - values[t]
                gae = delta
            else:
                delta = buf['rewards'][t] + self.gamma * values[t+1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns
        
    def update_policy(self):
        all_s, all_a, all_lp, all_adv, all_ret = [], [], [], [], []
        
        for tl_id, buf in self.agent_buffers.items():
            if len(buf['states']) == 0:
                continue
            adv, ret = self._compute_gae_for_agent(tl_id)
            all_s.extend(buf['states'])
            all_a.extend(buf['actions'])
            all_lp.extend(buf['log_probs'])
            all_adv.extend(adv)
            all_ret.extend(ret)
            
        if len(all_s) == 0:
            return 0.0, 0.0
            
        states = torch.FloatTensor(np.array(all_s))
        actions = torch.LongTensor(all_a)
        old_lp = torch.FloatTensor(all_lp)
        advantages = torch.FloatTensor(all_adv)
        returns = torch.FloatTensor(all_ret)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n = len(all_s)
        bs = min(128, n)
        p_loss_total, v_loss_total = 0.0, 0.0
        
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, bs):
                end = min(start + bs, n)
                i = idx[start:end]
                
                logits, vals = self.model(states[i])
                
                # Apply mask to logits in the training batch
                masked_logits = logits.clone()
                # states[i, 6] chứa cờ can_switch (1.0 nếu được phép đổi, 0.0 nếu không)
                can_switch_batch = states[i, 6] > 0.5
                masked_logits[:, 1] = torch.where(can_switch_batch, masked_logits[:, 1], torch.tensor(-1e9, device=logits.device))
                
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_lp = dist.log_prob(actions[i])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_lp - old_lp[i])
                s1 = ratio * advantages[i]
                s2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages[i]
                
                pl = -torch.min(s1, s2).mean()
                vl = (returns[i] - vals.squeeze()).pow(2).mean()
                loss = pl + self.value_loss_coef * vl - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                p_loss_total += pl.item()
                v_loss_total += vl.item()
        
        for tl_id in self.agent_buffers:
            self.agent_buffers[tl_id] = {
                'states': [], 'actions': [], 'log_probs': [],
                'rewards': [], 'dones': [], 'values': []
            }
        return p_loss_total / self.epochs, v_loss_total / self.epochs


def train():
    env = MultiSumoEnv('data/grid2x2.sumocfg', use_gui=False, max_steps=500, min_green=10)
    
    states = env.reset()
    tl_ids = list(states.keys())
    STATE_SIZE = len(states[tl_ids[0]])
    ACTION_SIZE = 2
    
    print(f"=" * 80)
    print(f"HUẤN LUYỆN MULTI-AGENT PPO - PHIÊN BẢN PHỐI HỢP GIAO LỘ")
    print(f"=" * 80)
    print(f"Số Agent: {len(tl_ids)} ({tl_ids})")
    print(f"State Size: {STATE_SIZE} | Action Space: {ACTION_SIZE}")
    print(f"Min Green: {env.min_green}s")
    print(f"Topology: {dict(env.neighbors)}")
    print(f"=" * 80)
    
    agent = MultiAgentPPO(
        STATE_SIZE, ACTION_SIZE,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        epochs=6,
        hidden_size=128,
        max_grad_norm=0.5,
        value_loss_coef=0.5,
        entropy_coef=0.06   # Entropy cao hơn → phá vỡ đồng bộ cứng nhắc, khuyến khích thích ứng
    )
    
    NUM_EPISODES = 500
    UPDATE_EVERY = 25
    
    all_rewards = []
    best_avg = -float('inf')
    os.makedirs('models', exist_ok=True)
    
    for ep in range(NUM_EPISODES):
        states = env.reset()
        ep_rewards = {tl_id: 0.0 for tl_id in tl_ids}
        ep_switches = {tl_id: 0 for tl_id in tl_ids}
        step = 0
        
        agent.update_schedule(ep / NUM_EPISODES)
        
        while True:
            actions = {}
            for tl_id in tl_ids:
                actions[tl_id] = agent.select_action_for_agent(tl_id, states[tl_id])
                
            next_states, rewards, done, actual_actions, _ = env.step(actions)
            
            for tl_id in tl_ids:
                agent.store_reward_for_agent(tl_id, rewards[tl_id], done)
                ep_rewards[tl_id] += rewards[tl_id]
                if actual_actions[tl_id] == 1:
                    ep_switches[tl_id] += 1
                
            step += 1
            if step % UPDATE_EVERY == 0:
                agent.update_policy()
            if done:
                agent.update_policy()
                break
            states = next_states
            
        total = sum(ep_rewards.values())
        all_rewards.append(total)
        avg = np.mean(all_rewards[-10:])
        
        if len(all_rewards) >= 10 and avg > best_avg:
            best_avg = avg
            torch.save(agent.model.state_dict(), 'models/ppo_multi_traffic_best.pth')
        
        sw_str = "/".join(str(ep_switches[t]) for t in tl_ids)
        print(f"Ep {ep+1:3d}/{NUM_EPISODES} | R: {total:8.1f} ({', '.join(f'{t}:{ep_rewards[t]:.0f}' for t in tl_ids)}) | Sw: {sw_str} | Avg10: {avg:8.1f} | Best: {best_avg:8.1f}")
        
    env.close()
    torch.save(agent.model.state_dict(), 'models/ppo_multi_traffic.pth')
    
    # Đồ thị
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(all_rewards, alpha=0.2, color='steelblue', label='Raw')
    w20 = [np.mean(all_rewards[max(0,i-20):i+1]) for i in range(len(all_rewards))]
    ax1.plot(w20, lw=2, color='darkorange', label='MA-20')
    ax1.axhline(y=best_avg, color='green', ls='--', alpha=0.5, label=f'Best: {best_avg:.0f}')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('System Reward')
    ax1.set_title('Multi-Agent PPO - Phối hợp giao lộ')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    if len(all_rewards) >= 50:
        w50 = [np.mean(all_rewards[max(0,i-50):i+1]) for i in range(len(all_rewards))]
        ax2.plot(w50, lw=2, color='crimson', label='MA-50')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Reward (MA-50)')
        ax2.set_title('Xu hướng dài hạn')
        ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_multi_result.png', dpi=150)
    plt.close()
    
    print(f"\nHuấn luyện hoàn tất! Best Avg10: {best_avg:.1f}")
    print(f"Model: models/ppo_multi_traffic_best.pth")

if __name__ == '__main__':
    train()
