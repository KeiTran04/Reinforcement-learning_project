"""
Huấn luyện DQN baseline cho multi-agent traffic light control.
Parameter Sharing giống PPO để so sánh công bằng.

Usage:
    python train_dqn.py                                 # Train trên grid2x2
    python train_dqn.py --cfg data/grid2x2.sumocfg      # Custom config
    python train_dqn.py --episodes 400                   # Số episode
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from multi_sumo_env import MultiSumoEnv
from dqn_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='data/grid2x2.sumocfg')
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--save-dir', default='models')
    args = parser.parse_args()

    writer = SummaryWriter('runs/dqn_traffic')
    os.makedirs(args.save_dir, exist_ok=True)

    env = MultiSumoEnv(args.cfg, use_gui=False, max_steps=500, min_green=10)
    states = env.reset()
    tl_ids = list(states.keys())
    state_size = len(states[tl_ids[0]])
    action_size = 2

    print("=" * 80)
    print("HUẤN LUYỆN DQN - BASELINE SO SÁNH VỚI PPO")
    print("=" * 80)
    print(f"Số Agent: {len(tl_ids)} ({tl_ids})")
    print(f"State Size: {state_size} | Action Space: {action_size}")
    print(f"Topology: {dict(env.neighbors)}")
    print("=" * 80)

    agent = DQNAgent(
        state_size, action_size,
        hidden_size=128, lr=1e-3, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
        buffer_size=50000, batch_size=64,
        target_update_freq=100, tau=0.005,
    )

    UPDATE_EVERY = 20
    all_rewards = []
    best_avg = -float('inf')

    for ep in range(args.episodes):
        states = env.reset()
        ep_reward = {tl_id: 0.0 for tl_id in tl_ids}
        ep_switches = {tl_id: 0 for tl_id in tl_ids}
        step = 0

        while True:
            actions = {}
            for tl_id in tl_ids:
                can_switch = states[tl_id][6] > 0.5
                actions[tl_id] = agent.select_action(states[tl_id], can_switch)

            next_states, rewards, done, actual_actions, _ = env.step(actions)

            for tl_id in tl_ids:
                can_switch_next = next_states[tl_id][6] > 0.5
                agent.store_experience(
                    states[tl_id], actions[tl_id],
                    rewards[tl_id], next_states[tl_id],
                    done,
                )
                ep_reward[tl_id] += rewards[tl_id]
                if actual_actions[tl_id] == 1:
                    ep_switches[tl_id] += 1

            step += 1
            if step % UPDATE_EVERY == 0:
                loss = agent.update()
            if done:
                loss = agent.update()
                break
            states = next_states

        total = sum(ep_reward.values())
        all_rewards.append(total)
        avg = np.mean(all_rewards[-10:])

        if avg > best_avg:
            best_avg = avg
            agent.save(f'{args.save_dir}/dqn_traffic_best.pth')

        sw_str = "/".join(str(ep_switches[t]) for t in tl_ids)
        print(f"Ep {ep+1:3d}/{args.episodes} | R: {total:8.1f} ({', '.join(f'{t}:{ep_reward[t]:.0f}' for t in tl_ids)}) | Sw: {sw_str} | Avg10: {avg:8.1f} | Eps: {agent.epsilon:.3f} | Best: {best_avg:8.1f}")

        writer.add_scalar('DQN/TotalReward', total, ep)
        writer.add_scalar('DQN/Avg10Reward', avg, ep)
        writer.add_scalar('DQN/Epsilon', agent.epsilon, ep)
        if agent.losses:
            writer.add_scalar('DQN/Loss', np.mean(agent.losses[-10:]), ep)

    env.close()
    writer.close()
    agent.save(f'{args.save_dir}/dqn_traffic.pth')

    # Đồ thị
    plt.figure(figsize=(12, 5))
    plt.plot(all_rewards, alpha=0.3, color='steelblue', label='Raw')
    w10 = [np.mean(all_rewards[max(0,i-10):i+1]) for i in range(len(all_rewards))]
    plt.plot(w10, lw=2, color='darkorange', label='MA-10')
    plt.axhline(y=best_avg, color='green', ls='--', alpha=0.5, label=f'Best: {best_avg:.0f}')
    plt.xlabel('Episode')
    plt.ylabel('System Reward')
    plt.title('DQN - Multi-Agent Traffic Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_dqn_result.png', dpi=150)
    plt.close()

    print(f"\nHuấn luyện DQN hoàn tất! Best Avg10: {best_avg:.1f}")
    print(f"Model: {args.save_dir}/dqn_traffic_best.pth")


if __name__ == '__main__':
    train()
