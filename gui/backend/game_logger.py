"""Append-only JSONL logging for GUI games."""

import json
import os
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

    @staticmethod
    def _new_id(prefix):
        return f'{prefix}_{uuid.uuid4().hex}'

    def new_participant_id(self):
        return self._new_id('participant')

    def new_session_id(self):
        return self._new_id('session')

    def new_game_id(self):
        return self._new_id('game')

    def write_event(self, event_type, payload):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        event = {'event_type': event_type, 'timestamp': self.clock()}
        event.update(payload)
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            fh.write('\n')

    def log_action(self, room_id, game_id, participant_id, player_id,
                   is_human, action, legal_action_count, current_rank,
                   num_cards_left, timings, account_id=None):
        self.write_event('action', {
            'room_id': room_id,
            'game_id': game_id,
            'participant_id': participant_id,
            'account_id': account_id,
            'player_id': player_id,
            'is_human': bool(is_human),
            'action': action,
            'legal_action_count': legal_action_count,
            'current_rank': current_rank,
            'num_cards_left': list(num_cards_left),
            'timings': dict(timings or {}),
        })
