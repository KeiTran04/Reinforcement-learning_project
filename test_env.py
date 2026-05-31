from sumo_env import SumoEnv
import numpy as np

env = SumoEnv('data/simple.sumocfg', use_gui=True)
state = env.reset()
print(f"State ban đầu: {state}")
total_reward = 0

for step in range(200):
    import time
    time.sleep(0.08)  # Làm chậm mô phỏng để mắt người kịp nhìn
    action = np.random.choice([0, 1])  # Hành động ngẫu nhiên
    state, reward, done = env.step(action)
    total_reward += reward
    if step % 20 == 0:
        print(f"Step {step}: state={state}, reward={reward:.1f}")
    if done:
        break

print(f"Tổng reward (ngẫu nhiên): {total_reward:.1f}")
env.close()
