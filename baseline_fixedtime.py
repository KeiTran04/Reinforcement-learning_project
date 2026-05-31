"""
Baseline comparison: Fixed-time controller vs RL agent.
Chạy cùng môi trường, so sánh các metrics:
- Average waiting time
- Throughput (số xe đã qua)
- Số lần đổi đèn
- Average queue length

Usage:
    python baseline_fixedtime.py                    # Chạy baseline + RL
    python baseline_fixedtime.py --mode fixed        # Chỉ chạy fixed-time
    python baseline_fixedtime.py --mode rl           # Chỉ chạy RL
    python baseline_fixedtime.py --cfg data/grid2x2.sumocfg
"""
import argparse
import numpy as np
import torch
import traci
import time
from multi_sumo_env import MultiSumoEnv
from ppo_agent import PPO


def run_episode(env, policy_fn, is_rl=False):
    """Chạy 1 episode với policy cho trước, trả về dict metrics."""
    states = env.reset()
    tl_ids = list(states.keys())

    total_waiting = 0.0
    total_halting = 0
    total_switches = 0
    total_flow = 0
    steps = 0

    while True:
        actions = {}
        for tl_id in tl_ids:
            if is_rl:
                state_t = torch.FloatTensor(states[tl_id]).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy_fn.model(state_t)
                can_switch = states[tl_id][6] > 0.5
                if not can_switch:
                    logits = logits.clone()
                    logits[0, 1] = -1e9
                actions[tl_id] = logits.argmax(dim=1).item()
            else:
                actions[tl_id] = policy_fn(tl_id, steps, env)

        next_states, rewards, done, actual_actions, info = env.step(actions)

        # Thu thập metrics
        for tl_id in tl_ids:
            total_waiting += sum(traci.lane.getWaitingTime(l)
                                 for l in env.controlled_lanes[tl_id])
            total_halting += sum(traci.lane.getLastStepHaltingNumber(l)
                                 for l in env.controlled_lanes[tl_id])
            total_flow += info[tl_id]['flow']
            if actual_actions[tl_id] == 1:
                total_switches += 1

        steps += 1
        if done:
            break
        states = next_states

    return {
        'avg_waiting': total_waiting / max(steps * len(tl_ids), 1),
        'avg_halting': total_halting / max(steps * len(tl_ids), 1),
        'total_switches': total_switches,
        'total_flow': total_flow,
        'steps': steps,
    }


class FixedTimePolicy:
    """Điều khiển đèn theo chu kỳ cố định."""

    def __init__(self, switches_interval=3):
        self.interval = switches_interval
        self.counter = 0

    def __call__(self, tl_id, step, env):
        if step % self.interval == 0:
            return 1
        return 0


def compare(cfg_path, n_episodes=5, switches_interval=3):
    """So sánh fixed-time vs RL trên nhiều episode."""
    print("=" * 80)
    print("SO SÁNH BASELINE: FIXED-TIME vs RL AGENT")
    print("=" * 80)

    # --- Fixed-time ---
    print(f"\n[1/2] Đang chạy Fixed-Time (đổi đèn mỗi {switches_interval * 5}s)...")
    ft_metrics = []
    for ep in range(n_episodes):
        env = MultiSumoEnv(cfg_path, use_gui=False, max_steps=500, min_green=10)
        ft_policy = FixedTimePolicy(switches_interval)
        m = run_episode(env, ft_policy, is_rl=False)
        ft_metrics.append(m)
        env.close()
        print(f"  Ep {ep + 1}: avg_waiting={m['avg_waiting']:.2f}, "
              f"avg_halting={m['avg_halting']:.2f}, switches={m['total_switches']}, "
              f"flow={m['total_flow']}")

    # --- RL ---
    print(f"\n[2/2] Đang chạy RL Agent (model: models/ppo_multi_traffic_best.pth)...")
    rl_metrics = []
    for ep in range(n_episodes):
        env = MultiSumoEnv(cfg_path, use_gui=False, max_steps=500, min_green=10)
        states = env.reset()
        tl_ids = list(states.keys())
        state_size = len(states[tl_ids[0]])

        agent = PPO(state_size, 2)
        try:
            agent.model.load_state_dict(
                torch.load('models/ppo_multi_traffic_best.pth', weights_only=True))
        except Exception as e:
            print(f"  Không tải được model: {e}. Dùng random weights.")
        agent.model.eval()

        m = run_episode(env, agent, is_rl=True)
        rl_metrics.append(m)
        env.close()
        print(f"  Ep {ep + 1}: avg_waiting={m['avg_waiting']:.2f}, "
              f"avg_halting={m['avg_halting']:.2f}, switches={m['total_switches']}, "
              f"flow={m['total_flow']}")

    # --- Tổng hợp ---
    print("\n" + "=" * 80)
    print("KẾT QUẢ SO SÁNH (trung bình trên {} episode)".format(n_episodes))
    print("=" * 80)
    print(f"{'Metric':<25} {'Fixed-Time':>15} {'RL Agent':>15} {'Cải thiện':>15}")
    print("-" * 70)

    for key, label, better in [
        ('avg_waiting', 'Thời gian chờ TB', '↓'),
        ('avg_halting', 'Hàng chờ TB', '↓'),
        ('total_switches', 'Số lần đổi đèn', '↓'),
        ('total_flow', 'Lưu lượng xe', '↑'),
    ]:
        ft_val = np.mean([m[key] for m in ft_metrics])
        rl_val = np.mean([m[key] for m in rl_metrics])
        if ft_val != 0:
            change = (rl_val - ft_val) / abs(ft_val) * 100
        else:
            change = 0.0
        arrow = '🟢' if (better == '↓' and change < 0) or (better == '↑' and change > 0) else '🔴'
        print(f"{label:<25} {ft_val:>15.2f} {rl_val:>15.2f} {arrow} {change:>+6.1f}%")

    print("=" * 70)


def run_custom_episode(cfg_path, use_rl=True, use_gui=True):
    """Chạy 1 episode với GUI để quan sát trực quan."""
    env = MultiSumoEnv(cfg_path, use_gui=use_gui, max_steps=1000, gui_delay=100, min_green=10)
    states = env.reset()
    tl_ids = list(states.keys())

    if use_rl:
        state_size = len(states[tl_ids[0]])
        agent = PPO(state_size, 2)
        agent.model.load_state_dict(
            torch.load('models/ppo_multi_traffic_best.pth', weights_only=True))
        agent.model.eval()

    ft_policy = FixedTimePolicy(3)
    step = 0
    try:
        while True:
            actions = {}
            for tl_id in tl_ids:
                if use_rl:
                    state_t = torch.FloatTensor(states[tl_id]).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = agent.model(state_t)
                    can_switch = states[tl_id][6] > 0.5
                    if not can_switch:
                        logits = logits.clone()
                        logits[0, 1] = -1e9
                    actions[tl_id] = logits.argmax(dim=1).item()
                else:
                    actions[tl_id] = ft_policy(tl_id, step, env)

            states, rewards, done, actual_actions, info = env.step(actions)
            step += 1
            if done:
                break
    except Exception:
        pass
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compare', 'fixed', 'rl'], default='compare')
    parser.add_argument('--cfg', default='data/grid2x2.sumocfg')
    parser.add_argument('--gui', action='store_true', help='Hiện GUI')
    parser.add_argument('--n', type=int, default=5, help='Số episode')
    args = parser.parse_args()

    if args.mode == 'compare':
        compare(args.cfg, n_episodes=args.n)
    elif args.mode == 'fixed':
        run_custom_episode(args.cfg, use_rl=False, use_gui=args.gui)
    elif args.mode == 'rl':
        run_custom_episode(args.cfg, use_rl=True, use_gui=args.gui)
