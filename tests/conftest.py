"""Setup: mock traci module trước khi import code project."""
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# === Mock SUMO/TraCI hoàn chỉnh ===
def create_mock_traci():
    """Tạo mock traci với các method cần thiết cho cả topology + reward test."""
    mock = MagicMock()

    # --- junction ---
    mock.junction.getPosition.side_effect = lambda tl_id: {
        'A0': (0.0, 0.0),
        'A1': (0.0, 200.0),
        'B0': (200.0, 0.0),
        'B1': (200.0, 200.0),
        'C0': (0.0, 0.0),
        'C1': (0.0, 200.0),
        'C2': (0.0, 400.0),
        'D0': (200.0, 0.0),
        'D1': (200.0, 200.0),
        'D2': (200.0, 400.0),
        'single': (0.0, 0.0),
    }.get(tl_id, (0.0, 0.0))

    # --- simulation ---
    mock.simulation.getTime.return_value = 0.0

    # --- lane ---
    def mock_lane_get_halting(lane_id):
        data = {
            'A0_0': 3, 'A0_1': 5, 'A0_2': 1, 'A0_3': 7,
            'B0_0': 2, 'B0_1': 4,
            'sing_0': 2, 'sing_1': 3,
        }
        return data.get(lane_id, 0)

    mock.lane.getLastStepHaltingNumber.side_effect = mock_lane_get_halting
    mock.lane.getLastStepVehicleNumber.return_value = 5
    mock.lane.getWaitingTime.return_value = 15.0

    # --- trafficlight ---
    def mock_tl_get_phase(tl_id):
        phases = {'A0': 0, 'A1': 1, 'B0': 0, 'B1': 1, 'single': 0}
        return phases.get(tl_id, 0)

    mock.trafficlight.getPhase.side_effect = mock_tl_get_phase
    mock.trafficlight.getRedYellowGreenState.return_value = 'Ggrr'
    mock.trafficlight.getIDList.return_value = ['A0', 'A1', 'B0', 'B1']
    mock.trafficlight.getAllProgramLogics.return_value = [
        MagicMock(getPhases=lambda: [MagicMock(), MagicMock(), MagicMock(), MagicMock()])
    ]
    mock.trafficlight.getControlledLanes.return_value = ['A0_0', 'A0_1']

    def mock_get_controlled_links(tl_id):
        return [
            [('A0_0', 'dest1', 'A0')],
            [('A0_1', 'dest2', 'A0')],
        ]
    mock.trafficlight.getControlledLinks.side_effect = mock_get_controlled_links
    mock.trafficlight.setPhase = MagicMock()

    # --- close/start ---
    mock.close = MagicMock()
    mock.start = MagicMock()

    return mock


@pytest.fixture(autouse=True)
def mock_traci(monkeypatch):
    """Tự động mock traci cho mọi test."""
    mock = create_mock_traci()
    monkeypatch.setitem(sys.modules, 'traci', mock)
    return mock


@pytest.fixture
def env_2x2():
    """Tạo MultiSumoEnv với traci đã mock cho grid 2x2."""
    from multi_sumo_env import MultiSumoEnv
    env = MultiSumoEnv.__new__(MultiSumoEnv)
    env.tl_ids = ['A0', 'A1', 'B0', 'B1']
    env.controlled_lanes = {
        tid: ['{}_0'.format(tid), '{}_1'.format(tid)] for tid in env.tl_ids
    }
    env.time_since_switch = {tid: 10 for tid in env.tl_ids}
    env.prev_waiting = {tid: 0.0 for tid in env.tl_ids}
    env.signal_lanes = {tid: {0: ['{}_0'.format(tid)], 1: ['{}_1'.format(tid)]} for tid in env.tl_ids}
    env.skip_frames = {tid: 0 for tid in env.tl_ids}
    env.neighbors = {}
    env.positions = {}
    env.min_green = 10
    env.max_steps = 1000
    env.current_step = 0
    env.action_mode = 'extended'
    return env
