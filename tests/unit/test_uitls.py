import json
import pytest
from unittest.mock import patch, MagicMock
from utils import get_action_list


# ---------------------------------------------------------
# Helper: create a temporary JSON config file
# ---------------------------------------------------------

@pytest.fixture
def tmp_action_config(tmp_path):
    cfg_path = tmp_path / "action_config.json"

    config = {
        "accel": {
            "action": "Accelerate",
            "spec": {"param1": 10}
        },
        "brake": {
            "action": "Brake",
            "spec": {"param1": 20}
        }
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    return cfg_path


# ---------------------------------------------------------
# Test: get_action_list
# ---------------------------------------------------------

def test_get_action_list_calls_factory_and_sets_sim_dt(tmp_action_config):
    sim_dt = 0.1

    # Mock CommandFactory.create
    with patch("utils.CommandFactory") as mock_factory:
        mock_factory.create = MagicMock(side_effect=lambda name, spec: (name, spec))

        action_list = get_action_list(sim_dt, cfg=str(tmp_action_config))

        # Should create two actions
        assert len(action_list) == 2

        # Validate calls to CommandFactory.create
        expected_calls = [
            ("Accelerate", {"param1": 10, "sim_dt": sim_dt}),
            ("Brake", {"param1": 20, "sim_dt": sim_dt}),
        ]

        # Extract actual calls
        actual_calls = [call.args for call in mock_factory.create.call_args_list]

        assert actual_calls == expected_calls

        # Validate returned list structure
        assert action_list[0][0] == "Accelerate"
        assert action_list[1][0] == "Brake"

        # Ensure sim_dt was injected into each spec
        assert action_list[0][1]["sim_dt"] == sim_dt
        assert action_list[1][1]["sim_dt"] == sim_dt


# ---------------------------------------------------------
# Test: empty config file
# ---------------------------------------------------------

def test_get_action_list_empty_config(tmp_path):
    cfg_path = tmp_path / "empty.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    with patch("utils.CommandFactory.create") as mock_create:
        actions = get_action_list(0.1, cfg=str(cfg_path))

        assert actions == []
        mock_create.assert_not_called()
