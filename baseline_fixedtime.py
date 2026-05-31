"""
Baseline comparison: Fixed-time vs PPO vs DQN.
Chạy cùng môi trường, so sánh các metrics:
- Average waiting time
- Throughput (số xe đã qua)
- Số lần đổi đèn
- Average queue length

Usage:
    python baseline_fixedtime.py                            # So sánh cả 3
    python baseline_fixedtime.py --policies fixed,ppo       # Chỉ fixed + PPO
    python baseline_fixedtime.py --policies ppo,dqn         # Chỉ PPO + DQN
    python baseline_fixedtime.py --mode fixed --gui         # Xem 1 policy
"""
import argparse
import numpy as np
import torch
import traci
from multi_sumo_env import MultiSumoEnv
from ppo_agent import PPO
from dqn_agent import DQNAgent


# ====== Policy wrappers (interface chung) ======

class FixedTimePolicy:
    def __init__(self, switches_interval=3):
        self.interval = switches_interval

    def __call__(self, tl_id, step, env):
        return 1 if step % self.interval == 0 else 0

    def eval(self):
        pass

    def load(self, path):
        pass


class PPOPolicy:
    def __init__(self, state_size, model_path='models/ppo_multi_traffic_best.pth'):
        self.agent = PPO(state_size, 2)
        try:
            self.agent.model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"  [PPO] Đã tải model: {model_path}")
        except Exception as e:
            print(f"  [PPO] Không tải được model ({e}), dùng random weights.")
        self.agent.model.eval()

    def __call__(self, tl_id, step, env):
        # step không dùng cho RL, nhưng giữ interface
        return self._select_action(states_cache.get(tl_id))

    def set_states(self, states):
        global states_cache
        states_cache = states

    def _select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.agent.model(state_t)
        can_switch = state[6] > 0.5
        if not can_switch:
            logits = logits.clone()
            logits[0, 1] = -1e9
        return logits.argmax(dim=1).item()


class DQNPolicy:
    def __init__(self, state_size, model_path='models/dqn_traffic_best.pth'):
        self.agent = DQNAgent(state_size, 2)
        try:
            self.agent.load(model_path)
            print(f"  [DQN] Đã tải model: {model_path}")
        except Exception as e:
            print(f"  [DQN] Không tải được model ({e}), dùng random weights.")

    def __call__(self, tl_id, step, env):
        return self._select_action(states_cache.get(tl_id))

    def set_states(self, states):
        global states_cache
        states_cache = states

    def _select_action(self, state):
        can_switch = state[6] > 0.5
        return self.agent.select_action(state, can_switch)


states_cache = {}


# ====== Runner ======

def run_episode(env, policy_fn, name='policy'):
    """Chạy 1 episode, trả về metrics. Hỗ trợ RL policy qua states_cache."""
    global states_cache
    states = env.reset()
    tl_ids = list(states.keys())
    states_cache = states

    total_waiting = 0.0
    total_halting = 0
    total_switches = 0
    total_flow = 0
    steps = 0

    while True:
        actions = {}
        for tl_id in tl_ids:
            actions[tl_id] = policy_fn(tl_id, steps, env)

        next_states, rewards, done, actual_actions, info = env.step(actions)
        states_cache = next_states

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

    n_agents = len(tl_ids)
    return {
        'avg_waiting': total_waiting / max(steps * n_agents, 1),
        'avg_halting': total_halting / max(steps * n_agents, 1),
        'total_switches': total_switches,
        'total_flow': total_flow,
        'steps': steps,
    }


# ====== So sánh ======

POLICY_MAP = {
    'fixed': ('Fixed-Time', lambda s, cfg: FixedTimePolicy()),
    'ppo': ('PPO (MAPPO)', lambda s, cfg: PPOPolicy(s, f'models/{cfg}_multi_traffic_best.pth' if cfg != 'simple' else 'models/ppo_traffic_best.pth')),
    'dqn': ('DQN', lambda s, cfg: DQNPolicy(s, f'models/dqn_traffic_best.pth')),
}


def compare(cfg_path, policies=None, n_episodes=5, switches_interval=3):
    """So sánh nhiều policies."""
    config_name = cfg_path.split('/')[-1].replace('.sumocfg', '')
    if policies is None:
        policies = ['fixed', 'ppo', 'dqn']

    print("=" * 100)
    print(f"SO SÁNH BASELINE: {', '.join(p.upper() for p in policies)}")
    print(f"Môi trường: {cfg_path}")
    print("=" * 100)

    all_metrics = {}

    for p_name in policies:
        label, factory = POLICY_MAP[p_name]
        print(f"\n[{p_name.upper()}] {label}...")

        metrics = []
        for ep in range(n_episodes):
            env = MultiSumoEnv(cfg_path, use_gui=False, max_steps=500, min_green=10)
            states = env.reset()
            tl_ids = list(states.keys())
            state_size = len(states[tl_ids[0]])

            policy = factory(state_size, config_name)
            if hasattr(policy, 'set_states'):
                pass  # states_cache được cập nhật trong run_episode

            m = run_episode(env, policy, label)
            metrics.append(m)
            env.close()
            print(f"  Ep {ep + 1}: avg_wait={m['avg_waiting']:.1f}, "
                  f"avg_halting={m['avg_halting']:.1f}, "
                  f"switches={m['total_switches']}, flow={m['total_flow']}")

        all_metrics[p_name] = metrics

    # === Bảng so sánh ===
    print("\n" + "=" * 100)
    print(f"KẾT QUẢ SO SÁNH (trung bình {n_episodes} episodes)")
    print("=" * 100)

    # Header
    header = f"{'Metric':<25}"
    for p in policies:
        header += f" {POLICY_MAP[p][0]:>15}"
    header += f" {'Tốt nhất':>15}"
    print(header)
    print("-" * 100)

    for key, label, better_dir in [
        ('avg_waiting', 'Thời gian chờ TB', 'min'),
        ('avg_halting', 'Hàng chờ TB', 'min'),
        ('total_switches', 'Số lần đổi đèn', 'min'),
        ('total_flow', 'Lưu lượng xe', 'max'),
    ]:
        row = f"{label:<25}"
        vals = {}
        for p in policies:
            v = np.mean([m[key] for m in all_metrics[p]])
            vals[p] = v
            row += f" {v:>15.2f}"

        if better_dir == 'min':
            best_val = min(vals.values())
            best_p = [k for k, v in vals.items() if v == best_val][0]
        else:
            best_val = max(vals.values())
            best_p = [k for k, v in vals.items() if v == best_val][0]
        row += f" {POLICY_MAP[best_p][0]:>15}"

        print(row)

    # % improvement so với fixed-time
    if 'fixed' in policies:
        print("\n" + "-" * 60)
        print("% CẢI THIỆN SO VỚI FIXED-TIME")
        print("-" * 60)
        for p in policies:
            if p == 'fixed':
                continue
            print(f"\n{POLICY_MAP[p][0]}:")
            for key, label, better_dir in [
                ('avg_waiting', '  Thời gian chờ', 'min'),
                ('total_flow', '  Lưu lượng xe', 'max'),
            ]:
                ft_v = np.mean([m[key] for m in all_metrics['fixed']])
                rl_v = np.mean([m[key] for m in all_metrics[p]])
                if ft_v != 0:
                    change = (rl_v - ft_v) / abs(ft_v) * 100
                else:
                    change = 0.0
                arrow = '🟢' if (better_dir == 'min' and change < 0) or (better_dir == 'max' and change > 0) else '🔴'
                print(f"  {label}: {ft_v:.2f} → {rl_v:.2f}  {arrow} {change:>+6.1f}%")

    print("=" * 100)


def run_custom_episode(cfg_path, policy_name='fixed', use_gui=True):
    """Chạy 1 episode với GUI để quan sát."""
    _, factory = POLICY_MAP[policy_name]
    config_name = cfg_path.split('/')[-1].replace('.sumocfg', '')

    env = MultiSumoEnv(cfg_path, use_gui=use_gui, max_steps=1000, gui_delay=100, min_green=10)
    states = env.reset()
    tl_ids = list(states.keys())
    policy = factory(len(states[tl_ids[0]]), config_name)

    ft_policy = FixedTimePolicy(3)
    step = 0
    try:
        while True:
            actions = {}
            for tl_id in tl_ids:
                if policy_name in ('ppo', 'dqn'):
                    global states_cache
                    states_cache = states
                actions[tl_id] = policy(tl_id, step, env)

            states, rewards, done, actual_actions, info = env.step(actions)
            step += 1
            if done:
                break
    except Exception:
        pass
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies', default='fixed,ppo,dqn',
                        help='Các policy để so sánh, cách bởi dấu phẩy')
    parser.add_argument('--mode', choices=['compare', 'fixed', 'ppo', 'dqn'], default=None)
    parser.add_argument('--cfg', default='data/grid2x2.sumocfg')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--n', type=int, default=5, help='Số episode')
    args = parser.parse_args()

    if args.mode:
        # Chạy 1 policy với GUI
        run_custom_episode(args.cfg, args.mode, args.gui)
    else:
        policies = [p.strip() for p in args.policies.split(',')]
        compare(args.cfg, policies=policies, n_episodes=args.n)
