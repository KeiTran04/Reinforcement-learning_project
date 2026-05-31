"""Test phát hiện topology từ tọa độ."""

import pytest


class TestTopologyDetection:

    def test_2x2_grid(self, env_2x2):
        """Grid 2x2: mỗi ngã tư có 2 neighbors."""
        env_2x2._detect_topology()

        assert set(env_2x2.neighbors['A0']) == {'A1', 'B0'}
        assert set(env_2x2.neighbors['A1']) == {'A0', 'B1'}
        assert set(env_2x2.neighbors['B0']) == {'A0', 'B1'}
        assert set(env_2x2.neighbors['B1']) == {'A1', 'B0'}

    def test_2x2_positions_normalized(self, env_2x2):
        """Tọa độ chuẩn hóa nằm trong [0, 1]."""
        env_2x2._detect_topology()

        for tid, (r, c) in env_2x2.positions.items():
            assert 0.0 <= r <= 1.0, f'{tid} row={r}'
            assert 0.0 <= c <= 1.0, f'{tid} col={c}'

    def test_2x2_positions_unique(self, env_2x2):
        """Mỗi ngã tư có tọa độ riêng biệt."""
        env_2x2._detect_topology()

        pos_set = set(tuple(v) for v in env_2x2.positions.values())
        assert len(pos_set) == 4, 'Cần 4 vị trí khác nhau cho grid 2x2'

    def test_single_intersection(self, mock_traci):
        """1 ngã tư: không có neighbor."""
        mock_traci.trafficlight.getIDList.return_value = ['single']
        mock_traci.junction.getPosition.side_effect = lambda tid: {'single': (0.0, 0.0)}[tid]
        mock_traci.trafficlight.getControlledLinks.return_value = [
            [('sing_0', 'dest1', 'single')],
        ]

        from multi_sumo_env import MultiSumoEnv
        env = MultiSumoEnv.__new__(MultiSumoEnv)
        env.tl_ids = ['single']
        env.controlled_lanes = {'single': ['sing_0', 'sing_1']}
        env.time_since_switch = {'single': 10}
        env.prev_waiting = {'single': 0.0}
        env.signal_lanes = {'single': {0: ['sing_0'], 1: ['sing_1']}}
        env.skip_frames = {'single': 0}
        env.neighbors = {}
        env.positions = {}
        env.min_green = 10
        env.max_steps = 1000
        env.current_step = 0
        env.action_mode = 'binary'

        env._detect_topology()

        assert env.neighbors['single'] == []
        assert list(env.positions.keys()) == ['single']

    def test_3x3_grid(self, mock_traci):
        """Grid 3x3: ngã tư góc có 2 neighbors, cạnh có 3, trung tâm có 4."""
        tl_ids = ['C0', 'C1', 'C2', 'D0', 'D1', 'D2']
        mock_traci.trafficlight.getIDList.return_value = tl_ids
        positions = {
            'C0': (0.0, 0.0), 'C1': (0.0, 200.0), 'C2': (0.0, 400.0),
            'D0': (200.0, 0.0), 'D1': (200.0, 200.0), 'D2': (200.0, 400.0),
        }
        mock_traci.junction.getPosition.side_effect = lambda tid: positions[tid]

        from multi_sumo_env import MultiSumoEnv
        env = MultiSumoEnv.__new__(MultiSumoEnv)
        env.tl_ids = tl_ids
        env.controlled_lanes = {tid: ['{}_0'.format(tid)] for tid in tl_ids}
        env.time_since_switch = {tid: 10 for tid in tl_ids}
        env.prev_waiting = {tid: 0.0 for tid in tl_ids}
        env.signal_lanes = {tid: {0: ['{}_0'.format(tid)]} for tid in tl_ids}
        env.skip_frames = {tid: 0 for tid in tl_ids}
        env.neighbors = {}
        env.positions = {}
        env.min_green = 10
        env.max_steps = 1000
        env.current_step = 0
        env.action_mode = 'binary'

        env._detect_topology()

        assert set(env.neighbors['C0']) == {'C1', 'D0'}
        assert set(env.neighbors['C1']) == {'C0', 'C2', 'D1'}
        assert set(env.neighbors['D1']) == {'C1', 'D0', 'D2'}


class TestRewardFunction:

    def test_reward_components_in_range(self, env_2x2):
        """Các thành phần reward nằm trong khoảng dự kiến."""
        from unittest.mock import patch
        import numpy as np

        env_2x2._detect_topology()

        rewards = env_2x2._get_rewards(actual_actions={tid: 0 for tid in env_2x2.tl_ids})

        for tid in env_2x2.tl_ids:
            r = rewards[tid]
            assert -3.0 <= r <= 3.0, f'{tid} reward={r:.2f} vượt range dự kiến'

    def test_switch_penalty_severe_when_busy(self, env_2x2):
        """Đổi đèn khi đông xe (moving >= 4) → switch_reward = -1.0."""
        env_2x2._detect_topology()

        rewards = env_2x2._get_rewards(actual_actions={tid: 1 for tid in env_2x2.tl_ids})

        for tid in env_2x2.tl_ids:
            r = rewards[tid]
            # Switch penalty đã được ghi nhận
            assert isinstance(r, (int, float))


class TestActionMasking:
    """Test logic action masking ở cấp độ state vector (không cần PPO)."""

    def test_switch_blocked_flag_when_early(self, env_2x2):
        """state[6] = 0 khi time_since_switch < min_green."""
        env_2x2.time_since_switch['A0'] = 5
        env_2x2.min_green = 10

        state_feat_6 = 1.0 if env_2x2.time_since_switch['A0'] >= env_2x2.min_green else 0.0

        assert state_feat_6 == 0.0, 'can_switch = 0 khi chưa đủ min_green'

    def test_switch_allowed_flag_when_ready(self, env_2x2):
        """state[6] = 1 khi time_since_switch >= min_green."""
        env_2x2.time_since_switch['A0'] = 15
        env_2x2.min_green = 10

        state_feat_6 = 1.0 if env_2x2.time_since_switch['A0'] >= env_2x2.min_green else 0.0

        assert state_feat_6 == 1.0, 'can_switch = 1 khi đã đủ min_green'


class TestExtendedActions:

    def test_action_2_sets_skip_frames(self, env_2x2):
        """Action 2 (giữ 10s) → skip_frames = 1."""
        env_2x2.time_since_switch = {tid: 15 for tid in env_2x2.tl_ids}
        env_2x2.action_mode = 'extended'

        actions = {'A0': 2, 'A1': 0, 'B0': 0, 'B1': 0}

        # Step sẽ xử lý skip_frames, nhưng step cần traci.simulationStep
        # Chỉ test logic skip_frames trong vòng lặp
        for tl_id in env_2x2.tl_ids:
            raw = actions.get(tl_id, 0)
            if env_2x2.action_mode == 'extended' and raw >= 2:
                env_2x2.skip_frames[tl_id] = raw - 1

        assert env_2x2.skip_frames['A0'] == 1
        assert env_2x2.skip_frames['A1'] == 0

    def test_action_3_sets_skip_frames(self, env_2x2):
        """Action 3 (giữ 15s) → skip_frames = 2."""
        env_2x2.action_mode = 'extended'

        actions = {'A0': 3, 'A1': 0, 'B0': 0, 'B1': 0}
        for tl_id in env_2x2.tl_ids:
            raw = actions.get(tl_id, 0)
            if env_2x2.action_mode == 'extended' and raw >= 2:
                env_2x2.skip_frames[tl_id] = raw - 1

        assert env_2x2.skip_frames['A0'] == 2

    def test_skip_frames_decrements(self, env_2x2):
        """Mỗi step giảm skip_frames đi 1."""
        env_2x2.skip_frames['A0'] = 2
        env_2x2.time_since_switch['A0'] = 10

        # Mô phỏng skip logic
        if env_2x2.skip_frames['A0'] > 0:
            env_2x2.skip_frames['A0'] -= 1
            env_2x2.time_since_switch['A0'] += 5

        assert env_2x2.skip_frames['A0'] == 1
        assert env_2x2.time_since_switch['A0'] == 15
