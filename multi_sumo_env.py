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
    
    def __init__(self, cfg_path, use_gui=False, max_steps=1000, gui_delay=0, min_green=10):
        self.cfg_path = cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.gui_delay = gui_delay
        self.min_green = min_green
        self.current_step = 0
        self.tl_ids = []
        self.controlled_lanes = {}
        self.time_since_switch = {}
        self.prev_waiting = {}
        self.neighbors = {}          # tl_id -> [neighbor_ids]
        self.positions = {}          # tl_id -> (row_norm, col_norm)
        self.signal_lanes = {}       # tl_id -> {signal_idx: [lane_ids]}
        
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
        
        for tl_id in self.tl_ids:
            lanes = sorted(set(traci.trafficlight.getControlledLanes(tl_id)))
            self.controlled_lanes[tl_id] = lanes
            self.time_since_switch[tl_id] = self.min_green
            self.prev_waiting[tl_id] = 0.0
            
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
        """Tự động phát hiện vị trí lưới và hàng xóm từ TL ID (A0, A1, B0, B1...)."""
        self.neighbors = {tl_id: [] for tl_id in self.tl_ids}
        
        # Phân tích vị trí từ ID: chữ cái = hàng, số = cột
        raw_pos = {}
        rows, cols = set(), set()
        for tl_id in self.tl_ids:
            r = ord(tl_id[0]) - ord('A')
            c = int(tl_id[1:])
            raw_pos[tl_id] = (r, c)
            rows.add(r); cols.add(c)
        
        max_r = max(rows) if rows else 1
        max_c = max(cols) if cols else 1
        self.positions = {
            tl_id: (r / max(max_r, 1), c / max(max_c, 1))
            for tl_id, (r, c) in raw_pos.items()
        }
        
        # Hàng xóm = khoảng cách Manhattan = 1
        for tl_id, (r1, c1) in raw_pos.items():
            for other_id, (r2, c2) in raw_pos.items():
                if tl_id != other_id and abs(r1-r2) + abs(c1-c2) == 1:
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
        """Reward thông minh khuyến khích 3 hành vi mong muốn.
        
        Reward = waiting_penalty + delta_bonus + green_bonus + switch_reward + coord_bonus
        """
        rewards = {}
        
        for tl_id in self.tl_ids:
            green_l, red_l = self._get_green_red_lanes(tl_id)
            
            # === 1. Phạt thời gian chờ (giảm tỷ lệ để tránh áp đảo các reward khác) ===
            current_waiting = sum(traci.lane.getWaitingTime(l) for l in self.controlled_lanes[tl_id])
            waiting_penalty = -current_waiting / 30.0
            
            # === 2. Delta bonus (thưởng cải thiện) ===
            prev = self.prev_waiting.get(tl_id, current_waiting)
            delta_bonus = np.clip((prev - current_waiting) / 30.0, -2.0, 2.0)
            
            # === 3. Thưởng sử dụng đèn xanh hiệu quả (KÉO DÀI KHI ĐÔNG) ===
            moving_on_green = self._count_moving(green_l)
            green_bonus = min(moving_on_green * 0.6, 3.0)
            
            # === 4. Phạt/thưởng đổi đèn thích ứng (GIẢM KHI ÍT) ===
            switch_reward = 0.0
            if actual_actions and actual_actions.get(tl_id, 0) == 1:
                if moving_on_green >= 4:
                    # Đổi khi lane xanh rất đông → phạt cực nặng
                    switch_reward = -8.0
                elif moving_on_green >= 2:
                    # Đổi khi lane xanh vừa → phạt nặng
                    switch_reward = -4.0
                elif moving_on_green >= 1:
                    # Đổi khi có ít xe → phạt nhẹ
                    switch_reward = -1.5
                else:
                    # Đổi khi hoàn toàn vắng xe → không phạt (khuyến khích giải tỏa lane đỏ)
                    switch_reward = 0.0
            
            # === 5. Thưởng phối hợp (ĐỒNG PHA VỚI HÀNG XÓM) ===
            my_pg = traci.trafficlight.getPhase(tl_id) // 2
            coord_bonus = sum(
                0.3 for n in self.neighbors[tl_id]
                if traci.trafficlight.getPhase(n) // 2 == my_pg
            )
            
            reward = waiting_penalty + delta_bonus + green_bonus + switch_reward + coord_bonus
            rewards[tl_id] = reward
            self.prev_waiting[tl_id] = current_waiting
            
        return rewards
    
    def step(self, actions):
        """Thực hiện hành động, trả về (states, rewards, done, actual_actions, info)."""
        yellow_tls = []
        actual_actions = {}
        
        for tl_id in self.tl_ids:
            action = actions.get(tl_id, 0)
            cur_phase = traci.trafficlight.getPhase(tl_id)
            
            if action == 1 and cur_phase % 2 == 0 and self.time_since_switch[tl_id] >= self.min_green:
                n_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
                traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % n_phases)
                yellow_tls.append(tl_id)
                self.time_since_switch[tl_id] = 2
                actual_actions[tl_id] = 1
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
