"""Socket.IO contract tests for the GUI backend."""

import json

import pytest

from gui.backend import server
from gui.backend.game_logger import GameLogger

SESSION_HOST = 'session_' + 'a' * 32


def session_id(index):
    return f'session_{index:032x}'


@pytest.fixture(autouse=True)
def isolated_server(tmp_path, monkeypatch):
    server.rooms.clear()
    monkeypatch.setattr(
        server,
        'game_logger',
        GameLogger(log_dir=tmp_path,
                   clock=lambda: '2026-07-03T00:00:00Z'),
    )
    yield tmp_path
    server.rooms.clear()


def make_client():
    return server.socketio.test_client(server.app)


def find_event(client, name):
    for event in client.get_received():
        if event['name'] == name:
            return event['args'][0]
    return None


def event_names(client):
    return [event['name'] for event in client.get_received()]


def test_healthz_reports_service_ready():
    response = server.app.test_client().get('/healthz')

    assert response.status_code == 200
    assert response.get_json() == {
        'ok': True,
        'service': 'guandan-gui',
    }


def test_agent_status_endpoint_reports_runtime_metadata():
    response = server.app.test_client().get('/api/agents/status')

    assert response.status_code == 200
    payload = response.get_json()
    assert 'dmc' in payload['status']
    assert payload['status']['llm']['runtime_config_required'] is True


def test_feedback_endpoint_writes_anonymous_feedback_log(isolated_server):
    response = server.app.test_client().post('/api/feedback', json={
        'kind': 'suggestion',
        'message': '希望移动端按钮再大一点',
        'page': 'in_game',
        'roomId': 'ABC123',
        'playerId': 0,
        'context': {
            'url': 'https://guandan.aiwatch.icu/',
            'apiKey': 'sk-secret',
        },
    })

    assert response.status_code == 200
    assert response.get_json()['ok'] is True

    events = read_log_events(isolated_server)
    feedback = next(event for event in events
                    if event['event_type'] == 'feedback')
    assert feedback['kind'] == 'suggestion'
    assert feedback['message'] == '希望移动端按钮再大一点'
    assert feedback['page'] == 'in_game'
    assert feedback['room_id'] == 'ABC123'
    assert feedback['player_id'] == 0
    assert 'sk-secret' not in json.dumps(feedback, ensure_ascii=False)


def test_feedback_endpoint_rejects_empty_message():
    response = server.app.test_client().post('/api/feedback', json={
        'kind': 'suggestion',
        'message': '   ',
    })

    assert response.status_code == 400
    assert response.get_json()['message'] == '反馈内容不能为空。'


def create_started_room(human_ids=(0, 1, 2, 3), debug_enabled=False):
    clients = [make_client()]
    config = {
        'human_player_ids': list(human_ids),
        'debug_enabled': debug_enabled,
    }
    clients[0].emit('create_room', {
        'player_config': config,
        'sessionId': SESSION_HOST,
    })
    created = find_event(clients[0], 'room_created')
    clients[0].host_token = created['hostToken']
    clients[0].resume_token = created['resumeToken']
    room_id = created['roomId']

    for index, _seat in enumerate(human_ids[1:], start=1):
        client = make_client()
        client.emit('join_room', {
            'roomId': room_id,
            'sessionId': session_id(index),
        })
        clients.append(client)

    return room_id, clients


def current_legal_action(room_id):
    game = server.rooms[room_id]['game']
    current = game.current_player()
    return current, game.env.get_state(current)['actions'][0]


def trace_length(room_id):
    game = server.rooms[room_id]['game']
    return len(game.env.get_state(game.current_player()).get('trace', []))


def read_log_events(path):
    log_path = path / 'guandan_gui.jsonl'
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding='utf-8').splitlines()
    ]


def test_non_member_cannot_play_for_a_human_seat():
    room_id, _clients = create_started_room()
    current, action = current_legal_action(room_id)
    outsider = make_client()
    before = trace_length(room_id)

    outsider.emit('player_action', {
        'roomId': room_id,
        'playerId': current,
        'action': action,
    })

    assert find_event(outsider, 'error')['message'] == '无权操作该房间。'
    assert trace_length(room_id) == before


def test_room_member_cannot_play_for_another_human_seat():
    room_id, clients = create_started_room()
    current, action = current_legal_action(room_id)
    wrong_seat = next(seat for seat in range(4) if seat != current)
    wrong_client = clients[wrong_seat]
    before = trace_length(room_id)

    wrong_client.emit('player_action', {
        'roomId': room_id,
        'playerId': current,
        'action': action,
    })

    assert find_event(wrong_client, 'error')['message'] == '不能代替其他座位出牌。'
    assert trace_length(room_id) == before


def test_only_host_can_change_ai_speed():
    room_id, clients = create_started_room()
    host, non_host = clients[0], clients[1]

    non_host.emit('set_ai_speed', {'roomId': room_id, 'speed': 'fast'})

    assert find_event(non_host, 'error')['message'] == '只有房主可以修改房间设置。'
    assert server.rooms[room_id]['game'].ai_speed == 'normal'

    host.emit('set_ai_speed', {
        'roomId': room_id,
        'speed': 'fast',
        'hostToken': host.host_token,
    })

    assert server.rooms[room_id]['game'].ai_speed == 'fast'


def test_debug_mode_requires_debug_room_and_host():
    room_id, clients = create_started_room(debug_enabled=False)
    host, non_host = clients[0], clients[1]

    host.emit('set_debug_mode', {
        'roomId': room_id,
        'enabled': True,
        'hostToken': host.host_token,
    })

    assert find_event(host, 'error')['message'] == '该房间未开启调测权限。'
    assert server.rooms[room_id]['game'].debug_enabled is False

    debug_room_id, debug_clients = create_started_room(debug_enabled=True)
    debug_non_host = debug_clients[1]

    debug_non_host.emit('set_debug_mode', {
        'roomId': debug_room_id,
        'enabled': False,
    })

    assert find_event(debug_non_host, 'error')['message'] == \
        '只有房主可以修改房间设置。'
    assert server.rooms[debug_room_id]['game'].debug_enabled is True


def test_debug_state_is_only_sent_to_authorized_viewer():
    _room_id, clients = create_started_room(debug_enabled=True)

    host_started = find_event(clients[0], 'game_started')
    joined_started = find_event(clients[1], 'game_started')

    assert host_started['debug_state'] is not None
    assert joined_started['debug_state'] is None
    assert host_started['state']['debug_allowed'] is True
    assert host_started['state']['viewer_can_debug'] is True
    assert joined_started['state']['debug_allowed'] is True
    assert joined_started['state']['viewer_can_debug'] is False


def test_rejoin_with_same_resume_token_keeps_participant_id():
    room_id, clients = create_started_room()
    first_join = find_event(clients[1], 'joined_room')
    first_participant_id = first_join['participantId']
    for client in clients:
        client.get_received()
    clients[1].disconnect()

    rejoin = make_client()
    rejoin.emit('join_room', {
        'roomId': room_id,
        'sessionId': session_id(1),
        'resumeToken': first_join['resumeToken'],
    })
    second_join = find_event(rejoin, 'joined_room')

    assert second_join['participantId'] == first_participant_id


def test_spoofed_host_session_cannot_control_room():
    host = make_client()
    host.emit('create_room', {
        'sessionId': SESSION_HOST,
        'player_config': {
            'human_player_ids': [0, 1],
            'debug_enabled': True,
        },
    })
    room_id = find_event(host, 'room_created')['roomId']
    attacker = make_client()
    attacker.emit('join_room', {
        'roomId': room_id,
        'sessionId': SESSION_HOST,
    })
    received = attacker.get_received()
    joined = next(event['args'][0] for event in received
                  if event['name'] == 'joined_room')
    started = next(event['args'][0] for event in received
                   if event['name'] == 'game_started')
    assert joined['sessionId'] != SESSION_HOST
    assert started['debug_state'] is None

    attacker.emit('set_ai_speed', {'roomId': room_id, 'speed': 'fast'})

    assert find_event(attacker, 'error')['message'] == '只有房主可以修改房间设置。'
    assert server.rooms[room_id]['game'].ai_speed == 'normal'

    attacker.emit('set_debug_mode', {'roomId': room_id, 'enabled': False})

    assert find_event(attacker, 'error')['message'] == '只有房主可以修改房间设置。'
    assert server.rooms[room_id]['game'].debug_enabled is True


def test_rejoin_does_not_restart_started_room(isolated_server):
    room_id, clients = create_started_room()
    first_join = find_event(clients[1], 'joined_room')
    for client in clients:
        client.get_received()
    clients[1].disconnect()

    rejoin = make_client()
    rejoin.emit('join_room', {
        'roomId': room_id,
        'sessionId': session_id(1),
        'resumeToken': first_join['resumeToken'],
    })

    assert 'game_started' not in event_names(clients[0])
    assert find_event(rejoin, 'game_started') is not None

    events = read_log_events(isolated_server)
    assert [event['event_type'] for event in events].count('game_started') == 1


def test_room_created_log_uses_anonymous_whitelisted_config(isolated_server):
    client = make_client()
    client.emit('create_room', {
        'sessionId': 'alice@example.com|secret-token',
        'player_config': {
            'human_player_ids': [0],
            'agentTypes': {'1': 'random', '2': 'random', '3': 'random'},
            'debug_enabled': False,
            'llmConfig': {
                'apiKey': 'sk-secret',
                'baseUrl': 'https://api.example.com/v1',
                'model': 'model-a',
            },
            'nickname': 'alice@example.com',
            'token': 'secret-token',
        },
    })

    created_payload = find_event(client, 'room_created')
    room = server.rooms[created_payload['roomId']]
    assert 'sk-secret' not in json.dumps(room['config'], ensure_ascii=False)
    assert 'sk-secret' not in json.dumps(
        room['game'].player_config, ensure_ascii=False)

    events = read_log_events(isolated_server)
    created = next(event for event in events
                   if event['event_type'] == 'room_created')

    assert created['session_id'].startswith('session_')
    assert 'nickname' not in created
    assert created['player_config'] == {
        'human_player_ids': [0],
        'agentTypes': {'1': 'random', '2': 'random', '3': 'random'},
        'debug_enabled': False,
    }
    serialized = json.dumps(created, ensure_ascii=False)
    assert 'alice@example.com' not in serialized
    assert 'secret-token' not in serialized
    assert 'sk-secret' not in serialized


def test_state_emission_records_broadcast_timing():
    room_id, clients = create_started_room()
    for client in clients:
        client.get_received()

    server._emit_state_to_room(room_id)

    assert 'broadcast_ms' in server.rooms[room_id]['game'].last_timings


def test_ai_driver_records_total_loop_timing():
    host = make_client()
    host.emit('create_room', {
        'sessionId': SESSION_HOST,
        'player_config': {
            'human_player_ids': [0],
            'agentTypes': {'1': 'random', '2': 'random', '3': 'random'},
            'ai_speed': 'fast',
        },
    })
    room_id = find_event(host, 'room_created')['roomId']

    server._drive_ai(room_id)

    assert 'ai_advance_loop_ms' in server.rooms[room_id]['game'].last_timings
