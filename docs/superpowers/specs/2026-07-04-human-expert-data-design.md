# Human Expert Data Collection Design

## Goal

Collect high-value human Guandan decisions from the Web GUI, especially games
where a human team beats strong learned agents such as DanZero+ and
PerfectDan, then export those decisions as supervised training examples with a
`question` field for the decision state and a `sentence` field for the chosen
action rationale.

## Assumptions

- Public user-facing rooms should never expose hidden hands in training
  prompts. A training sample should represent what the human could reasonably
  know at decision time.
- Debug rooms may store richer replay data for developers, but those hidden
  fields must be marked separately and excluded from public-information
  `question` generation by default.
- The first implementation should collect and export data. It should not train
  a model, rank users, or build an account-based reputation system.
- Strong-model filtering can use configured GUI seat agent names and model
  paths, because the GUI already knows whether seats are `danzero_plus`,
  `perfectdan`, `base7`, `llm`, or human.
- Human consent and privacy should be handled at the GUI level with a clear
  notice before public deployment. Stored records must not include API keys,
  host tokens, resume tokens, raw contact information, or arbitrary profile
  text.

## Selected Approach

Use a two-layer data pipeline.

1. Append raw decision snapshots during the game.
2. At game end, filter the snapshots into expert samples only when the human
   team won against configured strong opponents.
3. Export training-ready JSONL from the filtered records.

This keeps the original evidence available for future re-export while keeping
the actual training data compact and stable.

## Alternatives Considered

### Store Only `question` And `sentence`

This is simple, but it loses state structure. If the prompt format changes, old
data cannot be regenerated without replaying the game. It also makes automated
quality filters harder.

### Store Full Perfect-Information State In Every Sample

This is useful for algorithm analysis, but unsafe as the default supervised
dataset. A model trained on hidden hands would learn decisions that human
players could not make in real public play.

### Selected Hybrid

Store structured snapshots first, then generate `question` and `sentence` as a
derived view. Perfect-information fields may exist only in a separate debug
section and must be excluded unless an export explicitly requests oracle data.

## Raw Decision Record

Each human action in the GUI should be recorded as one append-only JSONL event
with this shape:

```json
{
  "event_type": "decision_snapshot",
  "schema_version": 2,
  "timestamp": "2026-07-04T13:00:00Z",
  "room_id": "F726A3",
  "game_id": "game_xxx",
  "decision_id": "decision_xxx",
  "participant_id": "participant_xxx",
  "session_id": "session_xxx",
  "account_id": "optional_hashed_account_id",
  "player_id": 0,
  "team_id": 0,
  "teammate_id": 2,
  "is_human": true,
  "agent_types": {
    "0": "human",
    "1": "perfectdan",
    "2": "base7",
    "3": "danzero_plus"
  },
  "state": {
    "player_ids": [0, 1, 2, 3],
    "rank_list": ["2", "3"],
    "play_team": 0,
    "current_rank": "2",
    "wildcard": "H2",
    "current_player": 0,
    "current_hand": ["S3", "H3", "C3", "S4", "H4"],
    "num_cards_left": [20, 5, 12, 2],
    "greaterPos": 3,
    "greaterAction": ["Pair", "K", ["SK", "HK"]],
    "recent_trace": [
      [1, ["Single", "Q", ["DQ"]]],
      [2, ["Single", "A", ["SA"]]],
      [3, ["Pair", "K", ["SK", "HK"]]]
    ],
    "last_actions": {
      "0": null,
      "1": ["Single", "Q", ["DQ"]],
      "2": ["Single", "A", ["SA"]],
      "3": ["Pair", "K", ["SK", "HK"]]
    },
    "bomb_history": [],
    "finished_players": []
  },
  "legal_actions": [
    ["PASS", "PASS", "PASS"],
    ["Pair", "A", ["SA", "HA"]],
    ["Bomb", "6", ["S6", "H6", "C6", "D6"]]
  ],
  "chosen_action_index": 1,
  "chosen_action": ["Pair", "A", ["SA", "HA"]],
  "visibility": "public"
}
```

The `state` object should use values already exposed by the environment:

- `current_hand`
- `actions`
- `trace`
- `rank_list`
- `play_team`
- `greaterPos`
- `greaterAction`
- `num_cards_left`
- `last_actions`
- `bomb_history`
- `finished_players`

The raw record may also include a `debug_state` block in authorized debug rooms,
but the default exporter must ignore that block.

## Game Outcome Record

At game end, write one outcome event:

```json
{
  "event_type": "game_outcome",
  "schema_version": 2,
  "timestamp": "2026-07-04T13:10:00Z",
  "room_id": "F726A3",
  "game_id": "game_xxx",
  "winner_team": 0,
  "finished_players": [0, 2, 1, 3],
  "agent_types": {
    "0": "human",
    "1": "perfectdan",
    "2": "base7",
    "3": "danzero_plus"
  }
}
```

The exporter should join `decision_snapshot` records to `game_outcome` by
`game_id`.

## Expert Sample Filter

A raw decision becomes an expert sample only when all conditions are true:

- The actor is human.
- The actor's team equals `winner_team`.
- At least one opponent seat is a strong model. The initial strong-model set is
  `{"danzero_plus", "perfectdan"}`.
- The chosen action exists in `legal_actions`.
- The snapshot visibility is `public`, unless the export command explicitly
  requests oracle/debug data.

This filter intentionally keeps all winning human decisions, not just the final
move. Later exports can add stricter filters such as "opponent has three or
fewer cards", "bomb was available", or "human chose not to bomb".

## Training Sample Format

The first training export should write one JSON object per line:

```json
{
  "question": "Here is a Guandan card game...",
  "sentence": "Reasonable candidate actions are...",
  "chosen_action": ["Pair", "A", ["SA", "HA"]],
  "chosen_action_index": 1,
  "legal_actions": [
    ["PASS", "PASS", "PASS"],
    ["Pair", "A", ["SA", "HA"]],
    ["Bomb", "6", ["S6", "H6", "C6", "D6"]]
  ],
  "source": {
    "type": "gui_human_win_vs_strong_model",
    "game_id": "game_xxx",
    "decision_id": "decision_xxx",
    "visibility": "public"
  },
  "quality": {
    "human_team_won": true,
    "opponent_agents": ["perfectdan", "danzero_plus"],
    "criticality": "opponent_near_finish"
  }
}
```

`question` should include:

- Game name and role.
- Player id, team id, and teammate id.
- Team levels and current level card.
- Current hand.
- Remaining card counts for all players.
- Current greatest action and who played it.
- Recent trace.
- Legal actions with explicit indexes.
- Instruction to compare candidate actions and then choose the best action.

`sentence` should include:

- A short list of plausible candidate actions.
- A compact tactical comparison.
- The final chosen legal action index and action.

The first exporter may use deterministic heuristic rationales. It should not
pretend that the human typed the rationale. If the rationale is generated, mark
it as generated in internal metadata or keep the wording factual.

## Example Training Sample

```json
{
  "question": "Here is a Guandan card game. You are player 0 on team 0, and your teammate is player 2. Team 0 is currently at level 2, team 1 is at level 3, and the current level card is H2, which is the wild card. Your current hand is [S3, H3, C3, S4, H4, S6, H6, C6, D6, S8, H8, ST, HT, CT, DQ, SQ, SA, HA, SB, HR]. The remaining card counts for players [0, 1, 2, 3] are [20, 5, 12, 2]. The current greatest action is from player 3: ['Pair', 'K', ['SK', 'HK']]. Recent actions are [[1, ['Single', 'Q', ['DQ']]], [2, ['Single', 'A', ['SA']]], [3, ['Pair', 'K', ['SK', 'HK']]]]. Your legal actions are indexed as: 0: ['PASS', 'PASS', 'PASS']; 1: ['Pair', 'A', ['SA', 'HA']]; 2: ['Bomb', '6', ['S6', 'H6', 'C6', 'D6']]. First, select several reasonable candidate actions. Then consider the risk from opponents and your teammate's position. Finally, provide the best action.",
  "sentence": "Reasonable candidate actions are action 1, the pair of A, and action 2, the bomb of 6. Passing is risky because player 3 has only 2 cards left and may finish soon. The bomb can stop the opponent, but it spends a high-value resource. The pair of A beats the current pair of K with lower cost and keeps the bomb for later. Therefore, I choose action 1: ['Pair', 'A', ['SA', 'HA']].",
  "chosen_action": ["Pair", "A", ["SA", "HA"]],
  "chosen_action_index": 1,
  "legal_actions": [
    ["PASS", "PASS", "PASS"],
    ["Pair", "A", ["SA", "HA"]],
    ["Bomb", "6", ["S6", "H6", "C6", "D6"]]
  ],
  "source": {
    "type": "gui_human_win_vs_strong_model",
    "game_id": "game_xxx",
    "decision_id": "decision_xxx",
    "visibility": "public"
  },
  "quality": {
    "human_team_won": true,
    "opponent_agents": ["perfectdan", "danzero_plus"],
    "criticality": "opponent_near_finish"
  }
}
```

## Data Flow

1. A human joins a GUI room.
2. The backend starts a game and records room config with sanitized agent
   types.
3. When a human acts, the backend captures the state immediately before
   `env.step(action)`.
4. The backend writes `decision_snapshot`.
5. Existing action logging continues to write the compact `action` event.
6. When the game ends, the backend writes `game_outcome`.
7. A CLI exporter reads the JSONL log, joins snapshots with outcomes, applies
   the expert filter, and writes training JSONL.

## Error Handling

- If an action is missing from `legal_actions`, keep the raw event but skip it
  during training export.
- If a game has snapshots but no outcome, skip it during training export.
- If a record has unknown `schema_version`, skip it and count it in the export
  summary.
- If `agent_types` is missing, treat the game as not strong-model-qualified.

## Privacy And Safety

- Do not store API keys, host tokens, resume tokens, or raw browser storage.
- Do not place free-form user feedback inside training samples.
- If account ids are added, store a stable internal id or hash, not phone
  numbers, emails, or display names.
- Public dataset exports should default to `visibility = "public"`.
- Oracle/debug exports must require an explicit CLI flag and should write a
  different `source.type`.

## Testing Strategy

- Unit-test snapshot construction from a real environment state.
- Unit-test expert filtering with human win, human loss, missing outcome, and
  no strong opponent.
- Unit-test question generation to ensure legal action indexes match the raw
  legal action list.
- Unit-test that debug-only fields do not appear in public `question`.
- Integration-test GUI logging by simulating a human action and a completed
  game.

## Success Criteria

- Human GUI decisions are stored with enough state to regenerate prompts.
- Winning human decisions against DanZero+ or PerfectDan can be exported as
  `question`/`sentence` JSONL.
- The exporter never includes hidden hands in public samples.
- Existing GUI logging and tests continue to pass.
