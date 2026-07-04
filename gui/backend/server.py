"""Flask + Socket.IO server for the Guandan web GUI.

Run from the repository root::

    python -m gui.backend.server

Open http://localhost:5000 (or the LAN address printed on startup). One
player creates a room and configures the four seats; the room id is shared
with friends on the same network who join the remaining human seats.
"""

import logging
import os
import hmac
import re
import socket
import threading
import time
import uuid
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

from .agents import agent_runtime_status, available_agents
from .game_manager import Game
from .game_logger import GameLogger
from .state_adapter import sanitize_room_config

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
#             'host_sid': sid, 'lock': RLock, ...}
rooms = {}
SESSION_ID_RE = re.compile(r'^session_[0-9a-f]{32}$')
FEEDBACK_KINDS = {'suggestion', 'bug', 'client_error'}
SENSITIVE_FEEDBACK_KEYS = (
    'api_key', 'apikey', 'authorization', 'hosttoken', 'password',
    'resumetoken', 'secret', 'token',
)
FEEDBACK_CONTEXT_KEYS = (
    'url', 'path', 'userAgent', 'viewport', 'roomId', 'playerId',
    'component', 'message', 'stack',
)


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


def _session_id_from(data):
    session_id = (data or {}).get('sessionId')
    if isinstance(session_id, str) and SESSION_ID_RE.fullmatch(session_id):
        return session_id
    return game_logger.new_session_id()


def _new_token(prefix):
    return f'{prefix}_{uuid.uuid4().hex}'


def _safe_text(value, limit):
    if value is None:
        return ''
    return str(value).strip()[:limit]


def _safe_player_id(value):
    try:
        player_id = int(value)
    except (TypeError, ValueError):
        return None
    return player_id if 0 <= player_id <= 3 else None


def _is_sensitive_feedback_key(key):
    normalized = re.sub(r'[^a-z0-9]', '', str(key).lower())
    return any(marker in normalized for marker in SENSITIVE_FEEDBACK_KEYS)


def _sanitize_feedback_context(context):
    if not isinstance(context, dict):
        return {}
    safe = {}
    for key in FEEDBACK_CONTEXT_KEYS:
        if key not in context or _is_sensitive_feedback_key(key):
            continue
        value = context[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = _safe_text(value, 2000)
    return safe


def _token_from(data, key):
    token = (data or {}).get(key)
    return token if isinstance(token, str) and token.strip() else ''


def _same_token(left, right):
    return bool(left and right and hmac.compare_digest(left, right))


def _is_host(room, sid):
    return sid == room.get('host_sid')


def _emit_state_to_sid(room_id, sid, event='update_state'):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    seat = room['players'].get(sid)
    if seat is None:
        return
    is_host = _is_host(room, sid)
    can_debug = bool(room.get('debug_allowed') and is_host)
    payload = game.frontend_payload(
        viewer_player_id=seat,
        include_debug=game.debug_enabled and can_debug,
        debug_allowed=room.get('debug_allowed', False),
        viewer_can_debug=can_debug,
        viewer_is_host=is_host,
    )
    socketio.emit(event, {
        'state': payload['play_state'],
        'debug_state': payload['debug_state'],
        'current_player': game.current_player(),
        'viewer_player_id': seat,
        'room_id': room_id,
    }, to=sid)


def _emit_state_to_room(room_id, event='update_state'):
    room = rooms.get(room_id)
    if not room:
        return
    start = time.perf_counter()
    for sid, seat in list(room['players'].items()):
        _emit_state_to_sid(room_id, sid, event=event)
    room['game'].last_timings['broadcast_ms'] = round(
        (time.perf_counter() - start) * 1000, 3)


def _identity_for_player(room, player_id):
    for sid, seat in room['players'].items():
        if seat == player_id:
            return (
                room.get('participants', {}).get(sid),
                room.get('sessions', {}).get(sid),
            )
    return None, None


def _participants_snapshot(room):
    snapshot = []
    for session_id, seat in room.get('session_seats', {}).items():
        snapshot.append({
            'player_id': seat,
            'participant_id': room.get('session_participants', {}).get(
                session_id),
            'session_id': session_id,
            'account_id': None,
        })
    return sorted(snapshot, key=lambda item: item['player_id'])


def _room_for_request(room_id):
    room = rooms.get(room_id)
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return None
    if request.sid not in room['players']:
        emit('error', {'message': '无权操作该房间。'})
        return None
    return room


def _require_host(room, data):
    host_token = _token_from(data, 'hostToken')
    if _same_token(host_token, room.get('host_token')):
        room['host_sid'] = request.sid
        return True
    emit('error', {'message': '只有房主可以修改房间设置。'})
    return False


def _maybe_start_ai_driver(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    with room['lock']:
        game = room['game']
        if room.get('ai_driving') or game.is_over() or game.is_waiting_for_human():
            return
        room['ai_driving'] = True
    socketio.start_background_task(_drive_ai, room_id)


def _log_last_action(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    meta = game.last_action_meta
    if not meta:
        return
    player_id = meta['player_id']
    participant_id, session_id = _identity_for_player(room, player_id)
    if not participant_id:
        participant_id = f'ai_{player_id}'
        session_id = None
    state = game.env.get_state(game.current_player())
    game_logger.log_action(
        room_id=room_id,
        game_id=game.game_id,
        participant_id=participant_id,
        session_id=session_id,
        player_id=player_id,
        is_human=meta['is_human'],
        action=meta['action'],
        legal_action_count=meta['legal_action_count'],
        current_rank=getattr(game.env.game, 'cur_rank', 0),
        num_cards_left=state.get('num_cards_left', []),
        timings=game.last_timings,
    )


def _log_last_decision_snapshot(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    snapshot = game.last_decision_snapshot
    if not snapshot:
        return
    player_id = snapshot['player_id']
    participant_id, session_id = _identity_for_player(room, player_id)
    snapshot = dict(snapshot)
    snapshot.update({
        'room_id': room_id,
        'game_id': game.game_id,
        'participant_id': participant_id,
        'session_id': session_id,
        'account_id': None,
    })
    game_logger.log_decision_snapshot(snapshot)
    game.last_decision_snapshot = None


def _log_match_summary(room_id):
    room = rooms.get(room_id)
    if not room or room.get('summary_logged'):
        return
    game = room['game']
    if not game.is_over():
        return
    room['summary_logged'] = True
    result = list(getattr(game.env.game.round, 'result', []))
    finished_players = [p for p in result if p >= 0]
    game_logger.write_event('match_summary', {
        'room_id': room_id,
        'game_id': game.game_id,
        'winner_team': game.env.game.winner_team,
        'finished_players': finished_players,
        'participants': _participants_snapshot(room),
        'account_id': None,
    })
    game_logger.log_game_outcome(
        room_id=room_id,
        game_id=game.game_id,
        winner_team=game.env.game.winner_team,
        finished_players=finished_players,
        agent_types=game.agent_types,
    )


def _drive_ai(room_id):
    """Drive AI/auto turns according to the room speed mode."""
    loop_start = time.perf_counter()
    try:
        while True:
            room = rooms.get(room_id)
            if not room:
                return
            with room['lock']:
                game = room['game']
                if game.is_over() or game.is_waiting_for_human():
                    break
                speed = game.ai_speed

            delay = AI_SPEED_DELAYS.get(speed, AI_TURN_DELAY)
            if delay > 0:
                socketio.sleep(delay)

            room = rooms.get(room_id)
            if not room:
                return
            with room['lock']:
                game = room['game']
                if game.is_over() or game.is_waiting_for_human():
                    break
                if not game.step_one_ai():
                    break
                _log_last_action(room_id)
                if game.ai_speed != 'fast':
                    _emit_state_to_room(room_id)

        room = rooms.get(room_id)
        if room:
            with room['lock']:
                _emit_state_to_room(room_id)
                _log_match_summary(room_id)
    finally:
        room = rooms.get(room_id)
        if room:
            with room['lock']:
                room['game'].last_timings['ai_advance_loop_ms'] = round(
                    (time.perf_counter() - loop_start) * 1000, 3)
                room['ai_driving'] = False


# ----------------------------------------------------------------------
# HTTP routes (serve the built SPA)
# ----------------------------------------------------------------------

@app.route('/healthz')
def healthz():
    return jsonify({'ok': True, 'service': 'guandan-gui'})


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


@app.route('/api/agents/status')
def list_agent_status():
    return jsonify({'agents': available_agents(), 'status': agent_runtime_status()})


@app.post('/api/feedback')
def submit_feedback():
    data = request.get_json(silent=True) or {}
    message = _safe_text(data.get('message'), 1200)
    if not message:
        return jsonify({'ok': False, 'message': '反馈内容不能为空。'}), 400

    kind = _safe_text(data.get('kind'), 32)
    if kind not in FEEDBACK_KINDS:
        kind = 'suggestion'

    feedback_id = game_logger.new_feedback_id()
    game_logger.write_event('feedback', {
        'feedback_id': feedback_id,
        'kind': kind,
        'message': message,
        'page': _safe_text(data.get('page'), 64) or 'unknown',
        'room_id': _safe_text(data.get('roomId') or data.get('room_id'), 64),
        'player_id': _safe_player_id(data.get('playerId')),
        'participant_id': _safe_text(data.get('participantId'), 80),
        'context': _sanitize_feedback_context(data.get('context')),
        'account_id': None,
    })
    return jsonify({'ok': True, 'feedbackId': feedback_id})


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
            with room['lock']:
                if request.sid not in room['players']:
                    continue
                seat = room['players'].pop(request.sid)
                participant_id = room.get('participants', {}).pop(
                    request.sid, None)
                session_id = room.get('sessions', {}).pop(request.sid, None)
            leave_room(room_id)
            game_logger.write_event('player_disconnected', {
                'room_id': room_id,
                'game_id': room['game'].game_id,
                'participant_id': participant_id,
                'session_id': session_id,
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
        safe_config = sanitize_room_config(config)

        room_id = uuid.uuid4().hex[:6].upper()
        creator_seat = human_ids[0]
        participant_id = game_logger.new_participant_id()
        session_id = _session_id_from(data)
        resume_token = _new_token('resume')
        host_token = _new_token('host')
        debug_allowed = bool(safe_config.get('debug_enabled', False))
        rooms[room_id] = {
            'game': game,
            'players': {request.sid: creator_seat},
            'participants': {request.sid: participant_id},
            'sessions': {request.sid: session_id},
            'session_seats': {session_id: creator_seat},
            'session_participants': {session_id: participant_id},
            'session_resume_tokens': {session_id: resume_token},
            'config': safe_config,
            'host_sid': request.sid,
            'host_session_id': session_id,
            'host_token': host_token,
            'debug_allowed': debug_allowed,
            'summary_logged': False,
            'started': False,
            'ai_driving': False,
            'lock': threading.RLock(),
        }
        join_room(room_id)
        logger.info('Room %s created by sid %s (seat %s).',
                    room_id, request.sid, creator_seat)
        game_logger.write_event('room_created', {
            'room_id': room_id,
            'game_id': game.game_id,
            'participant_id': participant_id,
            'session_id': session_id,
            'player_id': creator_seat,
            'account_id': None,
            'player_config': safe_config,
        })
        emit('room_created', {
            'roomId': room_id,
            'playerId': creator_seat,
            'participantId': participant_id,
            'sessionId': session_id,
            'resumeToken': resume_token,
            'hostToken': host_token,
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

    with room['lock']:
        human_ids = room['config'].get('human_player_ids', [])
        taken = set(room['players'].values())
        session_id = _session_id_from(data)
        resume_token = _token_from(data, 'resumeToken')
        known_token = room.get('session_resume_tokens', {}).get(session_id)
        can_resume = _same_token(resume_token, known_token)
        if known_token and not can_resume:
            session_id = game_logger.new_session_id()
            known_token = None
        seat = None
        participant_id = None

        if can_resume:
            preferred = room.get('session_seats', {}).get(session_id)
            if preferred in human_ids and preferred not in taken:
                seat = preferred
                participant_id = room.get('session_participants', {}).get(
                    session_id)
                resume_token = known_token

        if seat is None:
            reserved = set()
            if room.get('started'):
                reserved = set(room.get('session_seats', {}).values())
            unavailable = taken | reserved
            seat = next((pid for pid in human_ids if pid not in unavailable),
                        None)
            if seat is None:
                emit('error', {'message': '房间已满。'})
                return
            participant_id = game_logger.new_participant_id()
            resume_token = _new_token('resume')
            room.setdefault('session_seats', {})[session_id] = seat
            room.setdefault('session_participants', {})[session_id] = \
                participant_id
            room.setdefault('session_resume_tokens', {})[session_id] = \
                resume_token

        if (session_id == room.get('host_session_id') and
                _same_token(_token_from(data, 'hostToken'),
                            room.get('host_token'))):
            room['host_sid'] = request.sid

        room['players'][request.sid] = seat
        room.setdefault('participants', {})[request.sid] = participant_id
        room.setdefault('sessions', {})[request.sid] = session_id
        already_started = room.get('started', False)
        is_host = request.sid == room.get('host_sid')

    join_room(room_id)
    logger.info('sid %s joined room %s as seat %s.', request.sid, room_id, seat)
    game_logger.write_event('player_joined', {
        'room_id': room_id,
        'game_id': room['game'].game_id,
        'participant_id': participant_id,
        'session_id': session_id,
        'player_id': seat,
        'account_id': None,
    })
    emit('joined_room', {
        'roomId': room_id,
        'playerId': seat,
        'participantId': participant_id,
        'sessionId': session_id,
        'resumeToken': resume_token,
        **({'hostToken': room.get('host_token')} if is_host else {}),
    })
    emit('player_joined', {'sid': request.sid, 'playerId': seat}, room=room_id)
    if already_started:
        with room['lock']:
            _emit_state_to_sid(room_id, request.sid, event='game_started')
    else:
        check_and_start_game(room_id)


def check_and_start_game(room_id):
    """Start once every human seat is occupied; broadcast the first state."""
    room = rooms.get(room_id)
    if not room:
        return
    with room['lock']:
        if room.get('started'):
            return
        human_ids = room['config'].get('human_player_ids', [])
        if len(room['players']) != len(human_ids):
            return
        room['started'] = True
        logger.info('All humans joined room %s; starting.', room_id)
        game_logger.write_event('game_started', {
            'room_id': room_id,
            'game_id': room['game'].game_id,
            'human_player_ids': human_ids,
            'participants': _participants_snapshot(room),
            'account_id': None,
        })
        _emit_state_to_room(room_id, event='game_started')
    # If the opening leader is somehow an AI/auto turn, animate it.
    _maybe_start_ai_driver(room_id)


@socketio.on('player_action')
def handle_player_action(data):
    data = data or {}
    room_id = data.get('roomId')
    room = _room_for_request(room_id)
    if not room:
        return
    with room['lock']:
        game = room['game']
        player_id = room['players'].get(request.sid)
        if data.get('playerId') is not None and data.get('playerId') != player_id:
            emit('error', {'message': '不能代替其他座位出牌。'})
            return
        try:
            game.perform_action(player_id, data.get('action'))
        except ValueError as exc:
            emit('error', {'message': str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception('player_action failed')
            emit('error', {'message': f'服务器错误: {exc}'})
            return

        _log_last_decision_snapshot(room_id)
        _log_last_action(room_id)
        _emit_state_to_room(room_id)
        _log_match_summary(room_id)
    _maybe_start_ai_driver(room_id)


@socketio.on('set_ai_speed')
def handle_set_ai_speed(data):
    data = data or {}
    room_id = data.get('roomId')
    room = _room_for_request(room_id)
    if not room:
        return
    with room['lock']:
        if not _require_host(room, data):
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
            'participant_id': room.get('participants', {}).get(request.sid),
            'session_id': room.get('sessions', {}).get(request.sid),
            'account_id': None,
        })
        _emit_state_to_room(room_id)


@socketio.on('set_debug_mode')
def handle_set_debug_mode(data):
    data = data or {}
    room_id = data.get('roomId')
    room = _room_for_request(room_id)
    if not room:
        return
    with room['lock']:
        if not room.get('debug_allowed'):
            emit('error', {'message': '该房间未开启调测权限。'})
            return
        if not _require_host(room, data):
            return
        room['game'].set_debug_enabled(bool(data.get('enabled')))
        game_logger.write_event('debug_mode_changed', {
            'room_id': room_id,
            'game_id': room['game'].game_id,
            'debug_enabled': room['game'].debug_enabled,
            'participant_id': room.get('participants', {}).get(request.sid),
            'session_id': room.get('sessions', {}).get(request.sid),
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
