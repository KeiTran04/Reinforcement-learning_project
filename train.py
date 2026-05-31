import matplotlib.pyplot as plt
from sumo_env import SumoEnv
from ppo_agent import PPO
import numpy as np
import torch
import os

def train():
    env = SumoEnv('data/simple.sumocfg', use_gui=False, max_steps=1000)

    # Chạy thử 1 lần để biết kích thước state
    state = env.reset()
    STATE_SIZE = len(state)
    ACTION_SIZE = 2  # 0=giữ, 1=đổi

    agent = PPO(
        STATE_SIZE, ACTION_SIZE,
        lr=2e-4,           # Learning rate thấp hơn → học ổn định hơn
        gamma=0.99,         # Discount factor
        gae_lambda=0.95,    # GAE lambda → cân bằng bias/variance
        eps_clip=0.2,       # PPO clip range
        epochs=4,           # Ít epoch hơn → tránh overfit trên batch nhỏ
        hidden_size=128,    # Mạng lớn hơn → học pattern phức tạp hơn
        max_grad_norm=0.5,  # Gradient clipping
        value_loss_coef=0.5,
        entropy_coef=0.01
    )

    NUM_EPISODES = 300     # Nhiều episode hơn để AI có thời gian hội tụ
    UPDATE_EVERY = 50      # Thu thập nhiều dữ liệu hơn trước mỗi lần update

    all_rewards = []
    best_avg_reward = -float('inf')

    # Đảm bảo thư mục models tồn tại
    os.makedirs('models', exist_ok=True)

    for ep in range(NUM_EPISODES):
        state = env.reset()
        ep_reward = 0
        step = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_reward(reward, done)

            ep_reward += reward
            step += 1

            # Cập nhật AI định kỳ - thu thập đủ dữ liệu mới update
            if step % UPDATE_EVERY == 0:
                agent.update()

            if done:
                # Cập nhật lần cuối nếu còn dữ liệu
                if len(agent.states) > 0:
                    agent.update()
                break

            state = next_state

        all_rewards.append(ep_reward)
        avg = np.mean(all_rewards[-10:])

        # Lưu model tốt nhất
        if len(all_rewards) >= 10 and avg > best_avg_reward:
            best_avg_reward = avg
            torch.save(agent.model.state_dict(), 'models/ppo_traffic_best.pth')

        print(f"Episode {ep+1}/{NUM_EPISODES} | Reward: {ep_reward:.1f} | "
              f"Avg10: {avg:.1f} | Best Avg10: {best_avg_reward:.1f}")

    env.close()

    # Lưu model cuối cùng
    torch.save(agent.model.state_dict(), 'models/ppo_traffic.pth')

    # Vẽ đồ thị
    plt.figure(figsize=(12, 5))
    plt.plot(all_rewards, alpha=0.3, color='steelblue', label='Reward mỗi episode')
    # Đường trung bình trượt
    window = 10
    avg_rewards = [np.mean(all_rewards[max(0, i - window):i + 1]) for i in range(len(all_rewards))]
    plt.plot(avg_rewards, linewidth=2, color='darkorange', label=f'Trung bình {window} ep')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Quá trình học của AI (Phiên bản cải tiến)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_result.png', dpi=150)
    plt.show()
    print(f"\nĐã lưu model cuối vào models/ppo_traffic.pth")
    print(f"Đã lưu model tốt nhất vào models/ppo_traffic_best.pth (Avg10: {best_avg_reward:.1f})")

if __name__ == '__main__':
    train()
