import os
import sys
import traci
import numpy as np

class MultiSumoEnv:
    """Môi trường Multi-Agent RL cho điều khiển đèn giao thông thông minh.
    
    3 tính năng cốt lõi:
    1. KÉO DÀI ĐÈN XANH KHI ĐÔNG XE: green_utilization_bonus thưởng khi xe đang lưu thông,
       switch_penalty nặng hơn khi cắt ngang luồng xe đông → AI tự học giữ xanh lâu hơn.
    2. GIẢM XANH KHI ÍT XE: switch_penalty nhẹ khi lane xanh vắng xe → AI tự học đổi sớm.
    3. PHỐI HỢP GIAO LỘ: neighbor pressure + phase alignment trong state cho AI thấy
       tình hình hàng xóm; coordination_bonus thưởng khi đèn xanh đồng pha → tạo "sóng xanh".
    """
    
    MAX_QUEUE = 15.0
    MAX_WAIT_TIME = 200.0
    
    def __init__(self, cfg_path, use_gui=False, max_steps=1000, gui_delay=0, min_green=10, action_mode='binary'):
        self.cfg_path = cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.gui_delay = gui_delay
        self.min_green = min_green
        self.action_mode = action_mode  # 'binary' (2 actions) or 'extended' (4 actions)
        self.current_step = 0
        self.tl_ids = []
        self.controlled_lanes = {}
        self.time_since_switch = {}
        self.prev_waiting = {}
        self.neighbors = {}          # tl_id -> [neighbor_ids]
        self.positions = {}          # tl_id -> (row_norm, col_norm)
        self.signal_lanes = {}       # tl_id -> {signal_idx: [lane_ids]}
        self.skip_frames = {}        # tl_id -> skip count for extended actions
        
    def reset(self):
        try:
            traci.close()
        except:
            pass
            
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        cmd = [sumo_binary, '-c', self.cfg_path, '--no-warnings', '--start']
        if self.use_gui and self.gui_delay > 0:
            cmd += ['--delay', str(self.gui_delay)]
            
        traci.start(cmd)
        
        self.tl_ids = sorted(traci.trafficlight.getIDList())
        self.current_step = 0
        self.controlled_lanes = {}
        self.time_since_switch = {}
        self.prev_waiting = {}
        self.signal_lanes = {}
        self.skip_frames = {}
        
        for tl_id in self.tl_ids:
            lanes = sorted(set(traci.trafficlight.getControlledLanes(tl_id)))
            self.controlled_lanes[tl_id] = lanes
            self.time_since_switch[tl_id] = self.min_green
            self.prev_waiting[tl_id] = 0.0
            self.skip_frames[tl_id] = 0
            
            # Cache ánh xạ tín hiệu -> lane (gọi 1 lần duy nhất)
            links = traci.trafficlight.getControlledLinks(tl_id)
            self.signal_lanes[tl_id] = {}
            for idx, link_group in enumerate(links):
                lane_set = set()
                for link in link_group:
                    lane_set.add(link[0])  # incoming lane
                self.signal_lanes[tl_id][idx] = list(lane_set)
        
        self._detect_topology()
        
        for _ in range(10):
            traci.simulationStep()
            self.current_step += 1
            
        return self._get_states()
    
    def _detect_topology(self):
        """Tự động phát hiện vị trí và hàng xóm từ tọa độ giao lộ (không phụ thuộc quy tắc đặt tên)."""
        self.neighbors = {tl_id: [] for tl_id in self.tl_ids}
        
        # Lấy tọa độ thực từ SUMO
        raw_pos = {}
        for tl_id in self.tl_ids:
            x, y = traci.junction.getPosition(tl_id)
            raw_pos[tl_id] = (x, y)
        
        # Chuẩn hóa vị trí về [0, 1]
        xs = [p[0] for p in raw_pos.values()]
        ys = [p[1] for p in raw_pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)
        self.positions = {
            tl_id: ((x - min_x) / range_x, (y - min_y) / range_y)
            for tl_id, (x, y) in raw_pos.items()
        }
        
        # Hàng xóm: khoảng cách Manhattan gần nhất
        n_tls = len(self.tl_ids)
        if n_tls <= 1:
            return
            
        # Tính cell size (khoảng cách giữa 2 giao lộ liền kề)
        dists = []
        for i, (tl_a, (x1, y1)) in enumerate(raw_pos.items()):
            for tl_b, (x2, y2) in list(raw_pos.items())[i+1:]:
                dists.append(abs(x1 - x2) + abs(y1 - y2))
        cell = min(dists) * 1.5 if dists else 100.0
        
        for tl_id, (x1, y1) in raw_pos.items():
            for other_id, (x2, y2) in raw_pos.items():
                if tl_id != other_id:
                    if abs(x1 - x2) + abs(y1 - y2) < cell:
                        self.neighbors[tl_id].append(other_id)
    
    def _get_green_red_lanes(self, tl_id):
        """Trả về (green_lanes, red_lanes) dựa trên trạng thái đèn hiện tại."""
        state_str = traci.trafficlight.getRedYellowGreenState(tl_id)
        green, red = set(), set()
        for idx, lanes in self.signal_lanes[tl_id].items():
            if idx < len(state_str):
                if state_str[idx] in ('G', 'g'):
                    green.update(lanes)
                elif state_str[idx] == 'r':
                    red.update(lanes)
        return green, red
    
    def _count_moving(self, lanes):
        """Đếm số xe đang di chuyển (không đứng yên) trên tập lane."""
        return sum(
            max(0, traci.lane.getLastStepVehicleNumber(l) - traci.lane.getLastStepHaltingNumber(l))
            for l in lanes
        ) if lanes else 0
    
    def _count_halting(self, lanes):
        """Đếm số xe đứng yên trên tập lane."""
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes) if lanes else 0
        
    def _get_states(self):
        """Xây dựng state vector phong phú cho mỗi ngã tư.
        
        State gồm 13 features:
        [0..3]  Hàng chờ mỗi lane (chuẩn hóa /MAX_QUEUE)
        [4]     Pha đèn hiện tại (chuẩn hóa)
        [5]     Thời gian đã giữ pha (chuẩn hóa)
        [6]     Cờ được phép đổi đèn (0/1)
        [7]     Áp lực: (queue_đỏ - queue_xanh) / MAX_QUEUE
        [8]     Mức sử dụng đèn xanh: xe đang chạy trên lane xanh
        [9]     Áp lực trung bình hàng xóm
        [10]    Mức đồng pha với hàng xóm (0~1)
        [11-12] Vị trí lưới (hàng, cột chuẩn hóa)
        """
        # Tính trước áp lực và pha cho tất cả TL (cần cho neighbor info)
        all_pressure = {}
        all_phase_group = {}
        
        for tl_id in self.tl_ids:
            green_l, red_l = self._get_green_red_lanes(tl_id)
            g_q = self._count_halting(green_l)
            r_q = self._count_halting(red_l)
            all_pressure[tl_id] = np.clip((r_q - g_q) / self.MAX_QUEUE, -1.0, 1.0)
            all_phase_group[tl_id] = traci.trafficlight.getPhase(tl_id) // 2
        
        states = {}
        for tl_id in self.tl_ids:
            state = []
            green_l, red_l = self._get_green_red_lanes(tl_id)
            
            # [0..3] Hàng chờ chuẩn hóa
            for lane in self.controlled_lanes[tl_id]:
                q = traci.lane.getLastStepHaltingNumber(lane)
                state.append(min(q / self.MAX_QUEUE, 1.0))
            
            # [4] Pha đèn chuẩn hóa
            phase = traci.trafficlight.getPhase(tl_id)
            n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
            state.append(phase / max(n_phases - 1, 1))
            
            # [5] Thời gian đã giữ pha (chuẩn hóa, cap ở 1.0)
            elapsed = self.time_since_switch.get(tl_id, self.min_green)
            state.append(min(elapsed / (self.min_green * 3), 1.0))
            
            # [6] Cờ được phép đổi
            state.append(1.0 if elapsed >= self.min_green else 0.0)
            
            # [7] Áp lực riêng
            state.append(all_pressure[tl_id])
            
            # [8] Mức sử dụng đèn xanh (xe đang chạy / dung lượng)
            moving = self._count_moving(green_l)
            capacity = max(len(green_l) * 3, 1)
            state.append(min(moving / capacity, 1.0))
            
            # [9] Áp lực trung bình hàng xóm
            n_pressures = [all_pressure[n] for n in self.neighbors[tl_id]]
            state.append(np.mean(n_pressures) if n_pressures else 0.0)
            
            # [10] Mức đồng pha với hàng xóm
            my_pg = all_phase_group[tl_id]
            if self.neighbors[tl_id]:
                align = np.mean([1.0 if all_phase_group[n] == my_pg else 0.0
                                 for n in self.neighbors[tl_id]])
                state.append(align)
            else:
                state.append(0.0)
            
            # [11-12] Vị trí lưới
            row, col = self.positions[tl_id]
            state.append(row)
            state.append(col)
            
            states[tl_id] = np.array(state, dtype=np.float32)
        return states
        
    def _get_rewards(self, actual_actions=None):
        """Reward thông minh - các thành phần được chuẩn hóa về cùng scale [-1, 1].
        
        Reward = waiting_penalty + delta_bonus + green_bonus + switch_reward + coord_bonus
        
        Mỗi thành phần được đưa về scale ~1.0 để không thành phần nào áp đảo quá trình học.
        """
        rewards = {}
        
        for tl_id in self.tl_ids:
            green_l, red_l = self._get_green_red_lanes(tl_id)
            
            # === 1. Phạt hàng chờ: tanh đưa về [-1, 0] ===
            n_halting = self._count_halting(self.controlled_lanes[tl_id])
            waiting_penalty = -np.tanh(n_halting / 5.0)
            
            # === 2. Delta bonus: thưởng khi hàng chờ giảm ===
            prev_halting = self.prev_waiting.get(tl_id, n_halting)
            delta_halting = prev_halting - n_halting
            delta_bonus = np.clip(delta_halting / 5.0, -1.0, 1.0)
            
            # === 3. Thưởng luồng xanh: tỉ lệ xe đang chạy trên lane xanh ===
            moving_on_green = self._count_moving(green_l)
            green_bonus = np.clip(moving_on_green / 5.0, 0.0, 1.0)
            
            # === 4. Phạt đổi đèn thích ứng: scale [-1, 0] ===
            switch_reward = 0.0
            if actual_actions and actual_actions.get(tl_id, 0) == 1:
                # Giảm mức phạt để scale cân bằng với các thành phần khác
                if moving_on_green >= 4:
                    switch_reward = -1.0   # rất đông
                elif moving_on_green >= 2:
                    switch_reward = -0.5   # vừa
                elif moving_on_green >= 1:
                    switch_reward = -0.2   # ít
                else:
                    switch_reward = 0.0    # vắng hoàn toàn
            
            # === 5. Thưởng phối hợp: scale [0, 1] ===
            my_pg = traci.trafficlight.getPhase(tl_id) // 2
            n_neighbors = len(self.neighbors[tl_id])
            coord_bonus = sum(
                0.3 for n in self.neighbors[tl_id]
                if traci.trafficlight.getPhase(n) // 2 == my_pg
            ) / max(n_neighbors, 1)
            
            reward = waiting_penalty + delta_bonus + green_bonus + switch_reward + coord_bonus
            rewards[tl_id] = reward
            self.prev_waiting[tl_id] = n_halting
            
        return rewards
    
    def step(self, actions):
        """Thực hiện hành động, trả về (states, rewards, done, actual_actions, info).
        
        Actions (phụ thuộc action_mode):
          'binary':   0=giữ 5s,  1=đổi (có đèn vàng)
          'extended': 0=đổi,     1=giữ 5s,  2=giữ 10s,  3=giữ 15s
        """
        yellow_tls = []
        actual_actions = {}
        n_switches = 0
        
        for tl_id in self.tl_ids:
            # Extended: skip frames (action đã được commit từ trước)
            if self.skip_frames.get(tl_id, 0) > 0:
                self.skip_frames[tl_id] -= 1
                self.time_since_switch[tl_id] += 5
                actual_actions[tl_id] = 0
                continue
            
            raw = actions.get(tl_id, 0)
            cur_phase = traci.trafficlight.getPhase(tl_id)
            allowed = self.time_since_switch[tl_id] >= self.min_green
            
            if self.action_mode == 'extended':
                # Action 0=đổi, 1=giữ5s, 2=giữ10s, 3=giữ15s
                if raw == 0 and cur_phase % 2 == 0 and allowed:
                    # Switch
                    n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
                    traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % n_phases)
                    yellow_tls.append(tl_id)
                    self.time_since_switch[tl_id] = 2
                    actual_actions[tl_id] = 1
                    n_switches += 1
                else:
                    if raw >= 2:
                        self.skip_frames[tl_id] = raw - 1  # 2→1 skip, 3→2 skip
                    self.time_since_switch[tl_id] += 5
                    actual_actions[tl_id] = 0
            else:
                # Binary: 0=giữ, 1=đổi
                if raw == 1 and cur_phase % 2 == 0 and allowed:
                    n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
                    traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % n_phases)
                    yellow_tls.append(tl_id)
                    self.time_since_switch[tl_id] = 2
                    actual_actions[tl_id] = 1
                    n_switches += 1
                else:
                    self.time_since_switch[tl_id] += 5
                    actual_actions[tl_id] = 0
                
        for _ in range(3):
            traci.simulationStep()
            self.current_step += 1
            
        for tl_id in yellow_tls:
            cur_phase = traci.trafficlight.getPhase(tl_id)
            n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
            traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % n_phases)
                
        for _ in range(2):
            traci.simulationStep()
            self.current_step += 1
        
        # Tính info bổ sung cho test
        info = {}
        for tl_id in self.tl_ids:
            green_l, _ = self._get_green_red_lanes(tl_id)
            info[tl_id] = {
                'flow': self._count_moving(green_l),
                'green_time': self.time_since_switch[tl_id]
            }
            
        states = self._get_states()
        rewards = self._get_rewards(actual_actions)
        done = self.current_step >= self.max_steps
        
        return states, rewards, done, actual_actions, info
        
    def close(self):
        try:
            traci.close()
        except:
            pass
