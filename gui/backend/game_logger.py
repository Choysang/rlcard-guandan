"""Append-only JSONL logging for GUI games."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LOG_DIR = os.path.join('logs', 'gui')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class GameLogger:
    """Small JSONL writer used by the Socket.IO GUI backend."""

    def __init__(self, log_dir=None, clock=utc_now):
        self.log_dir = Path(log_dir or os.environ.get(
            'GUANDAN_GUI_LOG_DIR', DEFAULT_LOG_DIR))
        self.clock = clock
        self.path = self.log_dir / 'guandan_gui.jsonl'
        self._lock = threading.Lock()

    @staticmethod
    def _new_id(prefix):
        return f'{prefix}_{uuid.uuid4().hex}'

    def new_participant_id(self):
        return self._new_id('participant')

    def new_session_id(self):
        return self._new_id('session')

    def new_game_id(self):
        return self._new_id('game')

    def new_feedback_id(self):
        return self._new_id('feedback')

    def write_event(self, event_type, payload, schema_version=1):
        event = {
            'event_type': event_type,
            'schema_version': schema_version,
            'timestamp': self.clock(),
        }
        event.update(payload)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n'
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as fh:
                fh.write(line)

    def log_action(self, room_id, game_id, participant_id, player_id,
                   is_human, action, legal_action_count, current_rank,
                   num_cards_left, timings, account_id=None, session_id=None):
        self.write_event('action', {
            'room_id': room_id,
            'game_id': game_id,
            'participant_id': participant_id,
            'session_id': session_id,
            'account_id': account_id,
            'player_id': player_id,
            'is_human': bool(is_human),
            'action': action,
            'legal_action_count': legal_action_count,
            'current_rank': current_rank,
            'num_cards_left': list(num_cards_left),
            'timings': dict(timings or {}),
        })

    def log_decision_snapshot(self, snapshot):
        payload = dict(snapshot)
        payload.pop('event_type', None)
        payload.pop('schema_version', None)
        payload.pop('timestamp', None)
        self.write_event('decision_snapshot', payload, schema_version=2)

    def log_game_outcome(self, room_id, game_id, winner_team,
                         finished_players, agent_types):
        self.write_event('game_outcome', {
            'room_id': room_id,
            'game_id': game_id,
            'winner_team': winner_team,
            'finished_players': list(finished_players or []),
            'agent_types': {
                str(player_id): str(agent_type)
                for player_id, agent_type in dict(agent_types or {}).items()
            },
        }, schema_version=2)
