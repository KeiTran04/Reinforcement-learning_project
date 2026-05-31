import os
import sys
import traci
import numpy as np

class SumoEnv:
    """Môi trường giao tiếp giữa Python và SUMO."""
    
    def __init__(self, cfg_path, use_gui=False, max_steps=1000, gui_delay=0):
        self.cfg_path = cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.gui_delay = gui_delay  # Delay (ms) giữa mỗi bước trong GUI
        self.current_step = 0
        # Tìm ID ngã tư có đèn giao thông
        self.tl_id = None  # Sẽ được gán khi kết nối
        
    def reset(self):
        """Khởi động lại mô phỏng từ đầu, trả về state ban đầu."""
        # Đóng kết nối cũ nếu có
        try:
            traci.close()
        except:
            pass
        # Mở kết nối mới
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        cmd = [sumo_binary, '-c', self.cfg_path, '--no-warnings', '--start']
        if self.use_gui and self.gui_delay > 0:
            cmd += ['--delay', str(self.gui_delay)]
        traci.start(cmd)
        # Lấy ID đèn giao thông đầu tiên
        self.tl_id = traci.trafficlight.getIDList()[0]
        self.current_step = 0
        # Chạy vài bước để xe bắt đầu xuất hiện
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
        return self._get_state()
    
    def _get_state(self):
        """Đo lường tình trạng giao thông hiện tại."""
        # Lấy danh sách các lane đi vào ngã tư
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(lanes))
        state = []
        for lane in unique_lanes:
            # Số xe đang đứng chờ (halting = tốc độ < 0.1 m/s)
            queue = traci.lane.getLastStepHaltingNumber(lane)
            state.append(queue)
        # Thêm pha đèn hiện tại (0 hoặc 1)
        phase = traci.trafficlight.getPhase(self.tl_id)
        state.append(phase)
        return np.array(state, dtype=np.float32)
    
    def _get_reward(self):
        """Phần thưởng = âm của tổng số xe đang chờ (càng ít xe chờ càng tốt)."""
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(set(lanes))
        total_waiting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in unique_lanes
        )
        return -total_waiting
    
    def step(self, action):
        """
        Thực hiện 1 hành động và tiến mô phỏng.
        action: 0 = giữ nguyên, 1 = đổi đèn
        Trả về: (state_mới, reward, done)
        """
        if action == 1:
            current_phase = traci.trafficlight.getPhase(self.tl_id)
            # Chuyển sang pha tiếp theo
            next_phase = (current_phase + 1) % traci.trafficlight.getProgram(self.tl_id).__len__() \
                if hasattr(traci.trafficlight.getProgram(self.tl_id), '__len__') else (current_phase + 1) % 4
            traci.trafficlight.setPhase(self.tl_id, next_phase)
        
        # Chạy mô phỏng thêm vài bước (mỗi quyết định kéo dài 5 giây)
        for _ in range(5):
            traci.simulationStep()
            self.current_step += 1
        
        state = self._get_state()
        reward = self._get_reward()
        done = self.current_step >= self.max_steps
        
        return state, reward, done
    
    def close(self):
        traci.close()
