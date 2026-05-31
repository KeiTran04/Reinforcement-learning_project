"""
Hyperparameter Tuning với Optuna.
Tìm hyperparams tối ưu cho MAPPO trên bài toán điều khiển đèn giao thông.

Usage:
    pip install optuna
    python tune_hyperparams.py                          # Tuning đầy đủ
    python tune_hyperparams.py --n_trials 20            # Giới hạn số trial
    python tune_hyperparams.py --study_name ppo_v1      # Tên study
    python tune_hyperparams.py --resume                 # Tiếp tục study cũ
"""
import argparse
import numpy as np
from multi_sumo_env import MultiSumoEnv

try:
    import optuna
except ImportError:
    print("Vui lòng cài optuna: pip install optuna")
    exit(1)

from train_multi import MultiAgentPPO


# === Objective function ===
def create_objective(base_cfg='data/grid2x2.sumocfg', n_episodes=80):
    """Tạo objective function cho Optuna."""

    def objective(trial):
        # === Suggest hyperparams ===
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.90, 0.99)
        eps_clip = trial.suggest_float('eps_clip', 0.1, 0.3)
        epochs = trial.suggest_int('epochs', 4, 10)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        entropy_coef = trial.suggest_float('entropy_coef', 0.01, 0.15)
        value_loss_coef = trial.suggest_float('value_loss_coef', 0.3, 0.8)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)

        # === Khởi tạo env ===
        env = MultiSumoEnv(base_cfg, use_gui=False, max_steps=200, min_green=10)
        states = env.reset()
        tl_ids = list(states.keys())
        state_size = len(states[tl_ids[0]])
        action_size = 2

        # === Khởi tạo MultiAgentPPO ===
        agent = MultiAgentPPO(
            state_size, action_size,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            eps_clip=eps_clip, epochs=epochs,
            hidden_size=hidden_size, max_grad_norm=max_grad_norm,
            value_loss_coef=value_loss_coef, entropy_coef=entropy_coef,
        )

        # === Training nhanh ===
        update_every = 20
        all_rewards = []

        for ep in range(n_episodes):
            states = env.reset()
            ep_reward = 0.0
            step = 0
            agent.update_schedule(ep / n_episodes)

            while True:
                actions = {}
                for tl_id in tl_ids:
                    actions[tl_id] = agent.select_action_for_agent(tl_id, states[tl_id])

                next_states, rewards, done, actual_actions, _ = env.step(actions)

                for tl_id in tl_ids:
                    agent.store_reward_for_agent(tl_id, rewards[tl_id], done)
                    ep_reward += rewards[tl_id]

                step += 1
                if step % update_every == 0:
                    agent.update_policy()
                if done:
                    agent.update_policy()
                    break
                states = next_states

            all_rewards.append(ep_reward)

            if len(all_rewards) >= 5:
                interim = np.mean(all_rewards[-5:])
                trial.report(interim, ep)
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

        avg_last10 = np.mean(all_rewards[-min(10, len(all_rewards)):]) if all_rewards else -1000
        env.close()

        trial.set_user_attr('hidden_size', hidden_size)
        trial.set_user_attr('epochs', epochs)
        trial.set_user_attr('n_params', sum(p.numel() for p in agent.model.parameters()))

        return avg_last10

    return objective


def run_study(n_trials=30, study_name='ppo_traffic_tuning', resume=False, storage=None):
    """Chạy Optuna study."""
    # Load hoặc tạo study
    if resume and storage:
        study = optuna.load_study(study_name=study_name, storage=storage)
    elif resume:
        study = optuna.load_study(study_name=study_name)
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=resume,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

    print(f"{'='*80}")
    print(f"HYPERPARAMETER TUNING với Optuna")
    print(f"Study: {study_name} | Trials: {n_trials}")
    print(f"{'='*80}")

    # Objective
    objective = create_objective(n_episodes=80, eval_episodes=10)

    # Chạy optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # === Kết quả ===
    print(f"\n{'='*80}")
    print(f"KẾT QUẢ TUNING")
    print(f"{'='*80}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best reward: {study.best_trial.value:.2f}")
    print(f"\nBest hyperparams:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nBest attributes:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    # === So sánh với default ===
    print(f"\n{'='*80}")
    print(f"SO SÁNH VỚI DEFAULT HYPERPARAMS")
    print(f"{'='*80}")
    defaults = {
        'lr': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
        'epochs': 6, 'hidden_size': 128, 'entropy_coef': 0.06, 'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
    }
    print(f"{'Param':<20} {'Default':>15} {'Best':>15}")
    print("-" * 50)
    for key in defaults:
        d = defaults[key]
        b = study.best_trial.params.get(key, 'N/A')
        print(f"{key:<20} {str(d):>15} {str(b):>15}")

    # Xuất best params dạng config
    print(f"\n{'='*80}")
    print(f"CONFIG ĐỂ DÙNG TRONG TRAINING")
    print(f"{'='*80}")
    print(f"agent = MultiAgentPPO(")
    for key, value in study.best_trial.params.items():
        print(f"    {key}={repr(value)},")
    print(f")")

    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=30, help='Số trial')
    parser.add_argument('--study_name', default='ppo_traffic_tuning', help='Tên study')
    parser.add_argument('--resume', action='store_true', help='Tiếp tục study cũ')
    parser.add_argument('--storage', default=None, help='SQL storage URL (vd: sqlite:///tuning.db)')
    args = parser.parse_args()

    run_study(
        n_trials=args.n_trials,
        study_name=args.study_name,
        resume=args.resume,
        storage=args.storage,
    )
