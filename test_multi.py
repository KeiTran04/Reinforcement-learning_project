import torch
from multi_sumo_env import MultiSumoEnv
from ppo_agent import PPO
import numpy as np

def test():
    # Sử dụng model đã lưu
    env = MultiSumoEnv('data/grid2x2.sumocfg', use_gui=True, max_steps=1000, gui_delay=200, min_green=10)
    states = env.reset()
    tl_ids = list(states.keys())
    STATE_SIZE = len(states[tl_ids[0]])
    ACTION_SIZE = 2
    
    # Khởi tạo PPO agent (shared model)
    agent = PPO(STATE_SIZE, ACTION_SIZE)
    try:
        agent.model.load_state_dict(torch.load('models/ppo_multi_traffic_best.pth', weights_only=True))
        print("Đã tải thành công model Multi-Agent tốt nhất (models/ppo_multi_traffic_best.pth)")
    except Exception as e:
        print(f"Không thể tải model tốt nhất: {e}. Sử dụng mạng khởi tạo ngẫu nhiên.")
        
    agent.model.eval()
    
    total_rewards = {tl_id: 0.0 for tl_id in tl_ids}
    total_switches = {tl_id: 0 for tl_id in tl_ids}
    step = 0
    
    print(f"\nBắt đầu chạy thử nghiệm Multi-Agent (min_green={env.min_green}s)...")
    print("="*120)
    
    try:
        while True:
            probs = {}
            actions = {}
            for tl_id in tl_ids:
                state_t = torch.FloatTensor(states[tl_id]).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.model(state_t)
                
                # Action Masking: Nếu không được phép đổi (state[6] == 0.0), set logit của Action 1 = -1e9
                can_switch = states[tl_id][6] > 0.5
                if not can_switch:
                    logits = logits.clone()
                    logits[0, 1] = -1e9
                    
                # Tính xác suất chọn các hành động bằng Softmax
                p = torch.softmax(logits, dim=1).squeeze(0).numpy()
                probs[tl_id] = p
                action = logits.argmax(dim=1).item()
                actions[tl_id] = action
                
            states, rewards, done, actual_actions, info = env.step(actions)
            step += 1
            
            # Tính tổng reward hiện tại của hệ thống
            step_total = sum(rewards.values())
            for tl_id in tl_ids:
                total_rewards[tl_id] += rewards[tl_id]
                if actual_actions[tl_id] == 1:
                    total_switches[tl_id] += 1
            system_total = sum(total_rewards.values())
            
            # Format chuỗi hiển thị cho từng ngã tư
            info_parts = []
            for tl_id in tl_ids:
                act_str = 'ĐỔI' if actual_actions[tl_id] == 1 else 'giữ'
                p_switch = probs[tl_id][1]
                flow = info[tl_id]['flow']
                gt = info[tl_id]['green_time']
                info_parts.append(f"{tl_id}:{act_str}(P_sw:{p_switch:.2f}, F_gr:{flow}, T:{gt}s, R:{rewards[tl_id]:5.1f})")
                
            info_str = " | ".join(info_parts)
            print(f"Step {step:3d} | {info_str} | Step Total: {step_total:6.1f} | System Cum: {system_total:8.1f}")
            
            if done:
                break
    except Exception as e:
        # Xử lý khi người dùng đóng cửa sổ SUMO-GUI
        print(f"\nDừng mô phỏng (Kết nối bị đóng hoặc có lỗi: {e})")
        
    env.close()
    
    print("="*120)
    print("Test kết thúc!")
    for tl_id in tl_ids:
        print(f"Ngã tư {tl_id} | Tổng Reward: {total_rewards[tl_id]:8.1f} | Số lần đổi đèn: {total_switches[tl_id]}")
    print(f"Tổng Reward toàn hệ thống: {sum(total_rewards.values()):8.1f} ({step} bước)")
    print(f"Tổng số lần đổi đèn toàn hệ thống: {sum(total_switches.values())}")

if __name__ == '__main__':
    test()
