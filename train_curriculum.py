"""
Curriculum Learning: huấn luyện từ đơn giản đến phức tạp.
Stages: 1 ngã tư (simple) → 2x2 grid → 3x3 grid

Mỗi stage kế thừa weights từ stage trước.
Sử dụng extended action space (4 actions).
"""
import matplotlib.pyplot as plt
from multi_sumo_env import MultiSumoEnv
from ppo_agent import PPO, ActorCritic
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter


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
        lr = self.initial_lr * (1.0 - 0.9 * progress)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.entropy_coef = self.initial_entropy_coef * (1.0 - 0.5 * progress)

    def select_action_for_agent(self, tl_id, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)

        can_switch = state[6] > 0.5
        if not can_switch:
            logits = logits.clone()
            logits[0, 1] = -1e9

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(dim=1)
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
                masked_logits = logits.clone()
                can_switch_batch = states[i, 6] > 0.5
                masked_logits[:, 1] = torch.where(
                    can_switch_batch, masked_logits[:, 1], torch.tensor(-1e9, device=logits.device))

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


# ====== CURRICULUM STAGES ======
STAGES = [
    {
        'name': 'simple',
        'cfg': 'data/simple.sumocfg',
        'episodes': 200,
        'label': '1 ngã tư',
    },
    {
        'name': 'grid2x2',
        'cfg': 'data/grid2x2.sumocfg',
        'episodes': 300,
        'label': '2x2 grid (4 ngã tư)',
    },
    {
        'name': 'grid3x3',
        'cfg': 'data/grid3x3.sumocfg',
        'episodes': 400,
        'label': '3x3 grid (9 ngã tư)',
    },
]


def train_curriculum():
    writer = SummaryWriter('runs/curriculum_ppo')

    print("=" * 80)
    print("CURRICULUM LEARNING: MULTI-AGENT PPO")
    print("=" * 80)
    for i, stage in enumerate(STAGES):
        print(f"  Stage {i+1}: {stage['label']} ({stage['episodes']} episodes) - {stage['cfg']}")
    print("=" * 80)

    agent = None
    global_episode = 0

    for stage_idx, stage in enumerate(STAGES):
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx+1}/{len(STAGES)}: {stage['label']} ({stage['cfg']})")
        print(f"{'='*80}")

        env = MultiSumoEnv(stage['cfg'], use_gui=False,
                           max_steps=500, min_green=10,
                           action_mode='extended')

        states = env.reset()
        tl_ids = list(states.keys())
        STATE_SIZE = len(states[tl_ids[0]])
        ACTION_SIZE = 4  # extended: 0=đổi, 1=giữ5s, 2=giữ10s, 3=giữ15s

        print(f"  Số Agent: {len(tl_ids)} ({tl_ids})")
        print(f"  State Size: {STATE_SIZE} | Action Size: {ACTION_SIZE}")
        print(f"  Topology: {dict(env.neighbors)}")

        if agent is None:
            # Stage 1: tạo agent mới
            agent = MultiAgentPPO(
                STATE_SIZE, ACTION_SIZE,
                lr=3e-4, gamma=0.99, gae_lambda=0.95,
                eps_clip=0.2, epochs=6, hidden_size=128,
                max_grad_norm=0.5, value_loss_coef=0.5,
                entropy_coef=0.06,
            )
        else:
            # Stage 2+: kế thừa weights, tạo model mới nếu state_size thay đổi
            if agent.model.shared[0].in_features != STATE_SIZE:
                old_state_dict = agent.model.state_dict()
                agent.__init__(STATE_SIZE, ACTION_SIZE, lr=3e-4, gamma=0.99,
                               gae_lambda=0.95, eps_clip=0.2, epochs=6,
                               hidden_size=128, max_grad_norm=0.5,
                               value_loss_coef=0.5, entropy_coef=0.06)
                # Copy các weights tương thích (shared layers, actor, critic)
                new_state_dict = agent.model.state_dict()
                for key in old_state_dict:
                    if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
                        new_state_dict[key] = old_state_dict[key]
                agent.model.load_state_dict(new_state_dict)
                print(f"  Kế thừa weights từ stage trước ({sum(p.numel() for p in agent.model.parameters())} params)")

        NUM_EPISODES = stage['episodes']
        UPDATE_EVERY = 25

        all_rewards = []
        best_avg = -float('inf')

        for ep in range(NUM_EPISODES):
            states = env.reset()
            ep_rewards = {tl_id: 0.0 for tl_id in tl_ids}
            ep_switches = {tl_id: 0 for tl_id in tl_ids}
            step = 0

            progress = global_episode / sum(s['episodes'] for s in STAGES)
            agent.update_schedule(progress)

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

            if avg > best_avg:
                best_avg = avg
                torch.save(agent.model.state_dict(),
                           f'models/ppo_curriculum_{stage["name"]}_best.pth')

            sw_str = "/".join(str(ep_switches[t]) for t in tl_ids)
            print(f"[{stage['name']}] Ep {ep+1:3d}/{NUM_EPISODES} | "
                  f"R: {total:8.1f} ({', '.join(f'{t}:{ep_rewards[t]:.0f}' for t in tl_ids)}) | "
                  f"Sw: {sw_str} | Avg10: {avg:8.1f}")

            writer.add_scalar(f'Curriculum/{stage["name"]}/TotalReward', total, global_episode)
            writer.add_scalar(f'Curriculum/{stage["name"]}/Avg10Reward', avg, global_episode)
            for tl_id in tl_ids:
                writer.add_scalar(f'Curriculum/{stage["name"]}/{tl_id}_Reward',
                                  ep_rewards[tl_id], global_episode)
                writer.add_scalar(f'Curriculum/{stage["name"]}/{tl_id}_Switches',
                                  ep_switches[tl_id], global_episode)

            global_episode += 1

        # Lưu model cuối stage
        torch.save(agent.model.state_dict(), f'models/ppo_curriculum_{stage["name"]}.pth')
        env.close()
        print(f"  Kết thúc Stage {stage_idx+1}! Best Avg10: {best_avg:.1f}")

    writer.close()
    print(f"\n{'='*80}")
    print(f"CURRICULUM LEARNING HOÀN TẤT! ({global_episode} episodes)")
    print(f"{'='*80}")


if __name__ == '__main__':
    train_curriculum()
