"""GUI agent configuration tests."""

import os
import numpy as np
import pytest

from gui.backend import agents


def test_configured_weight_path_prefers_specific_env(monkeypatch):
    monkeypatch.setenv('GUANDAN_MODEL_DIR', '/models')
    monkeypatch.setenv('GUANDAN_DMC_MODEL_TAR', '/custom/dmc.tar')

    assert agents.configured_weight_path('dmc') == '/custom/dmc.tar'
    assert agents.configured_weight_path('danzero_plus') == os.path.normpath(
        '/models/danzero_plus/model.tar')


def test_agent_runtime_status_reports_non_secret_weight_metadata(monkeypatch):
    monkeypatch.setenv('GUANDAN_MODEL_DIR', '/models')

    status = agents.agent_runtime_status()

    assert status['llm']['requires'] == ['model', 'base_url', 'api_key']
    assert status['llm']['runtime_config_required'] is True
    assert 'apiKey' not in str(status)
    assert status['dmc']['path'] == os.path.normpath('/models/dmc/model.tar')
    assert status['danzero']['path'].endswith('q_network.ckpt')


def test_llm_config_requires_room_model_and_key():
    with pytest.raises(ValueError, match='模型名称、Base URL 和 API Key'):
        agents.build_agents({
            'human_player_ids': [0],
            'agentTypes': {'1': 'llm'},
            'llmConfig': {'apiKey': 'sk-test'},
        }, np.random.RandomState(1))


def test_llm_agent_uses_room_config(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAgent:
        def __init__(self, player_id, np_random, client):
            self.player_id = player_id
            self.np_random = np_random
            self.client = client

    monkeypatch.setattr(agents, 'LLMClient', FakeClient)
    monkeypatch.setattr(agents, 'LLMAgent', FakeAgent)

    built = agents.build_agents({
        'human_player_ids': [0, 2, 3],
        'agentTypes': {'1': 'llm'},
        'llmConfig': {
            'apiKey': ' sk-test ',
            'baseUrl': ' https://api.example.com/v1 ',
            'model': ' model-a ',
        },
    }, np.random.RandomState(1))

    assert isinstance(built[1], FakeAgent)
    assert captured == {
        'api_key': 'sk-test',
        'base_url': 'https://api.example.com/v1',
        'model': 'model-a',
    }


def test_missing_weighted_agent_file_fails_clearly(monkeypatch):
    monkeypatch.setattr(agents, 'configured_weight_path',
                        lambda _name: '/missing/model.tar')

    with pytest.raises(ValueError, match='无法加载 AI "dmc"'):
        agents.build_agents({
            'human_player_ids': [0, 2, 3],
            'agentTypes': {'1': 'dmc'},
        }, np.random.RandomState(1))
