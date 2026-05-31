from sumo_env import SumoEnv
from ppo_agent import PPO, ActorCritic
import torch
import numpy as np

def test():
    # gui_delay=200 → nhanh hơn, vẫn quan sát được
    env = SumoEnv('data/simple.sumocfg', use_gui=True, max_steps=1000, gui_delay=200)
    state = env.reset()
    STATE_SIZE = len(state)
    ACTION_SIZE = 2
    
    agent = PPO(STATE_SIZE, ACTION_SIZE)
    agent.model.load_state_dict(torch.load('models/ppo_traffic_best.pth', weights_only=True))
    agent.model.eval()
    
    total_reward = 0
    step = 0
    try:
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.model(state_t)
            action = logits.argmax(dim=1).item()
            
            state, reward, done = env.step(action)
            total_reward += reward
            step += 1
            print(f"Step {step:3d} | Action: {'đổi đèn  ' if action == 1 else 'giữ nguyên'} | Reward: {reward:6.1f} | Tổng: {total_reward:8.1f}")
            if done:
                break
    except Exception:
        print(f"\nSUMO đã đóng sau {step} bước.")
    
    print(f"Test xong! Tổng reward: {total_reward:.1f} ({step} bước)")
    try:
        env.close()
    except Exception:
        pass

if __name__ == '__main__':
    test()
