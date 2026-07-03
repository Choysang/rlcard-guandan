"""Flask + Socket.IO server for the Guandan web GUI.

Run from the repository root::

    python -m gui.backend.server

Open http://localhost:5000 (or the LAN address printed on startup). One
player creates a room and configures the four seats; the room id is shared
with friends on the same network who join the remaining human seats.
"""

import logging
import os
import socket
import threading
import uuid
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

from .agents import available_agents
from .game_manager import Game
from .game_logger import GameLogger

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('guandan.gui')

# Serve the built frontend (gui/frontend/dist) when present.
FRONTEND_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist'))
PORT = int(os.environ.get('GUANDAN_GUI_PORT', '5000'))
# Delay between animated AI turns, in seconds.
AI_TURN_DELAY = float(os.environ.get('GUANDAN_GUI_AI_DELAY', '0.8'))
AI_SPEED_DELAYS = {
    'fast': 0.0,
    'normal': AI_TURN_DELAY,
    'slow': float(os.environ.get('GUANDAN_GUI_AI_SLOW_DELAY', '1.6')),
}
# Socket.IO origins; '*' is convenient on a trusted LAN. Override with a
# comma-separated allow-list in production.
CORS_ORIGINS = os.environ.get('GUANDAN_GUI_CORS', '*')
_cors_origins = '*' if CORS_ORIGINS == '*' else [
    o.strip() for o in CORS_ORIGINS.split(',') if o.strip()]

app = Flask(__name__, static_folder=FRONTEND_DIST)
CORS(app, origins=_cors_origins)
socketio = SocketIO(app, cors_allowed_origins=_cors_origins,
                    async_mode='threading')
game_logger = GameLogger()

# room_id -> {'game': Game, 'players': {sid: seat}, 'config': {...},
#             'host_sid': sid}
rooms = {}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def get_local_ip():
    """Best-effort LAN IP for the startup banner."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return '127.0.0.1'


def _emit_state_to_room(room_id, event='update_state'):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    for sid, seat in list(room['players'].items()):
        payload = game.frontend_payload(viewer_player_id=seat)
        socketio.emit(event, {
            'state': payload['play_state'],
            'debug_state': payload['debug_state'],
            'current_player': game.current_player(),
            'viewer_player_id': seat,
            'room_id': room_id,
        }, to=sid)


def _participant_for(room, player_id):
    for sid, seat in room['players'].items():
        if seat == player_id:
            return room.get('participants', {}).get(sid)
    return None


def _log_last_action(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    meta = game.last_action_meta
    if not meta:
        return
    player_id = meta['player_id']
    participant_id = _participant_for(room, player_id) or f'ai_{player_id}'
    state = game.env.get_state(game.current_player())
    game_logger.log_action(
        room_id=room_id,
        game_id=game.game_id,
        participant_id=participant_id,
        player_id=player_id,
        is_human=meta['is_human'],
        action=meta['action'],
        legal_action_count=meta['legal_action_count'],
        current_rank=getattr(game.env.game, 'cur_rank', 0),
        num_cards_left=state.get('num_cards_left', []),
        timings=game.last_timings,
    )


def _log_match_summary(room_id):
    room = rooms.get(room_id)
    if not room or room.get('summary_logged'):
        return
    game = room['game']
    if not game.is_over():
        return
    room['summary_logged'] = True
    result = list(getattr(game.env.game.round, 'result', []))
    game_logger.write_event('match_summary', {
        'room_id': room_id,
        'game_id': game.game_id,
        'winner_team': game.env.game.winner_team,
        'finished_players': [p for p in result if p >= 0],
        'account_id': None,
    })


def _drive_ai(room_id):
    """Drive AI/auto turns according to the room speed mode."""
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    if game.ai_speed == 'fast':
        while not game.is_over() and not game.is_waiting_for_human():
            if room_id not in rooms:  # room may close mid-loop
                return
            if not game.step_one_ai():
                break
            _log_last_action(room_id)
        _emit_state_to_room(room_id)
        _log_match_summary(room_id)
        return

    delay = AI_SPEED_DELAYS.get(game.ai_speed, AI_TURN_DELAY)
    while not game.is_over() and not game.is_waiting_for_human():
        socketio.sleep(delay)
        if room_id not in rooms:  # room may close mid-loop
            return
        if not game.step_one_ai():
            break
        _log_last_action(room_id)
        _emit_state_to_room(room_id)
    _log_match_summary(room_id)


# ----------------------------------------------------------------------
# HTTP routes (serve the built SPA)
# ----------------------------------------------------------------------

@app.route('/')
def serve_index():
    index = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index):
        return send_from_directory(app.static_folder, 'index.html')
    return (
        '<h1>🃏 Guandan GUI server is running</h1>'
        f'<p>LAN address: http://{get_local_ip()}:{PORT}</p>'
        '<p>The frontend is not built yet. Run <code>npm run build</code> '
        'in <code>gui/frontend</code>, or start the Vite dev server with '
        '<code>npm run dev</code>.</p>'), 200


@app.route('/<path:path>')
def serve_static(path):
    full = os.path.join(app.static_folder, path)
    if os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/server-info')
def server_info():
    return jsonify({'ip': get_local_ip(), 'port': PORT,
                    'url': f'http://{get_local_ip()}:{PORT}'})


@app.route('/api/agents')
def list_agents():
    return jsonify({'agents': available_agents()})


# ----------------------------------------------------------------------
# Socket.IO events
# ----------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected: %s', request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected: %s', request.sid)
    for room_id, room in list(rooms.items()):
        if request.sid in room['players']:
            seat = room['players'].pop(request.sid)
            participant_id = room.get('participants', {}).pop(request.sid, None)
            leave_room(room_id)
            game_logger.write_event('player_disconnected', {
                'room_id': room_id,
                'game_id': room['game'].game_id,
                'participant_id': participant_id,
                'player_id': seat,
                'account_id': None,
            })
            emit('player_left', {'playerId': seat, 'sid': request.sid},
                 room=room_id)
            if not room['players']:
                rooms.pop(room_id, None)
                logger.info('Room %s closed (empty).', room_id)
            break


@socketio.on('create_room')
def handle_create_room(data):
    try:
        config = (data or {}).get('player_config', {})
        human_ids = config.get('human_player_ids', [])
        if not human_ids:
            emit('error', {'message': '创建房间失败：至少需要一名人类玩家。'})
            return

        seed = int(np.random.SeedSequence().generate_state(1)[0])
        game = Game(config, seed=seed)
        game.init_game()

        room_id = uuid.uuid4().hex[:6].upper()
        creator_seat = human_ids[0]
        participant_id = game_logger.new_participant_id()
        rooms[room_id] = {
            'game': game,
            'players': {request.sid: creator_seat},
            'participants': {request.sid: participant_id},
            'config': config,
            'host_sid': request.sid,
            'summary_logged': False,
        }
        join_room(room_id)
        logger.info('Room %s created by sid %s (seat %s).',
                    room_id, request.sid, creator_seat)
        game_logger.write_event('room_created', {
            'room_id': room_id,
            'game_id': game.game_id,
            'participant_id': participant_id,
            'player_id': creator_seat,
            'nickname': config.get('nickname', ''),
            'account_id': None,
            'player_config': config,
        })
        emit('room_created', {
            'roomId': room_id,
            'playerId': creator_seat,
            'participantId': participant_id,
        })
        check_and_start_game(room_id)
    except Exception as exc:  # noqa: BLE001 - report to the client
        logger.exception('create_room failed')
        emit('error', {'message': f'创建房间失败: {exc}'})


@socketio.on('join_room')
def handle_join_room(data):
    room_id = (data or {}).get('roomId')
    room = rooms.get(room_id)
    if not room:
        emit('error', {'message': '房间不存在。'})
        return

    human_ids = room['config'].get('human_player_ids', [])
    taken = set(room['players'].values())
    seat = next((pid for pid in human_ids if pid not in taken), None)
    if seat is None:
        emit('error', {'message': '房间已满。'})
        return

    join_room(room_id)
    participant_id = game_logger.new_participant_id()
    room['players'][request.sid] = seat
    room.setdefault('participants', {})[request.sid] = participant_id
    logger.info('sid %s joined room %s as seat %s.', request.sid, room_id, seat)
    game_logger.write_event('player_joined', {
        'room_id': room_id,
        'game_id': room['game'].game_id,
        'participant_id': participant_id,
        'player_id': seat,
        'account_id': None,
    })
    emit('joined_room', {
        'roomId': room_id,
        'playerId': seat,
        'participantId': participant_id,
    })
    emit('player_joined', {'sid': request.sid, 'playerId': seat}, room=room_id)
    check_and_start_game(room_id)


def check_and_start_game(room_id):
    """Start once every human seat is occupied; broadcast the first state."""
    room = rooms.get(room_id)
    if not room:
        return
    human_ids = room['config'].get('human_player_ids', [])
    if len(room['players']) != len(human_ids):
        return
    logger.info('All humans joined room %s; starting.', room_id)
    game_logger.write_event('game_started', {
        'room_id': room_id,
        'game_id': room['game'].game_id,
        'human_player_ids': human_ids,
        'account_id': None,
    })
    _emit_state_to_room(room_id, event='game_started')
    # If the opening leader is somehow an AI/auto turn, animate it.
    socketio.start_background_task(_drive_ai, room_id)


@socketio.on('player_action')
def handle_player_action(data):
    data = data or {}
    room_id = data.get('roomId')
    room = rooms.get(room_id)
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return
    game = room['game']
    try:
        game.perform_action(data.get('playerId'), data.get('action'))
    except ValueError as exc:
        emit('error', {'message': str(exc)})
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception('player_action failed')
        emit('error', {'message': f'服务器错误: {exc}'})
        return

    _log_last_action(room_id)
    _emit_state_to_room(room_id)
    _log_match_summary(room_id)
    socketio.start_background_task(_drive_ai, room_id)


@socketio.on('set_ai_speed')
def handle_set_ai_speed(data):
    data = data or {}
    room_id = data.get('roomId')
    room = rooms.get(room_id)
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return
    try:
        room['game'].set_ai_speed(data.get('speed'))
    except ValueError as exc:
        emit('error', {'message': str(exc)})
        return
    game_logger.write_event('ai_speed_changed', {
        'room_id': room_id,
        'game_id': room['game'].game_id,
        'speed': room['game'].ai_speed,
        'account_id': None,
    })
    _emit_state_to_room(room_id)


@socketio.on('set_debug_mode')
def handle_set_debug_mode(data):
    data = data or {}
    room_id = data.get('roomId')
    room = rooms.get(room_id)
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return
    room['game'].set_debug_enabled(bool(data.get('enabled')))
    game_logger.write_event('debug_mode_changed', {
        'room_id': room_id,
        'game_id': room['game'].game_id,
        'debug_enabled': room['game'].debug_enabled,
        'account_id': None,
    })
    _emit_state_to_room(room_id)


def _open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{PORT}')


def main():
    ip = get_local_ip()
    banner = '\n'.join([
        '',
        '=' * 60,
        '        Guandan Web GUI server',
        '=' * 60,
        f'  Local:  http://localhost:{PORT}',
        f'  LAN:    http://{ip}:{PORT}',
        '  One player creates a room; others join with the room id.',
        '  Ctrl+C to stop.',
        '=' * 60,
        '',
    ])
    # The Windows console may use a non-UTF-8 codec; never let logging
    # banners crash startup.
    try:
        print(banner)
    except UnicodeEncodeError:
        print(banner.encode('ascii', 'replace').decode('ascii'))
    if os.environ.get('GUANDAN_GUI_OPEN_BROWSER', 'false').lower() == 'true':
        threading.Thread(target=_open_browser, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=PORT,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
