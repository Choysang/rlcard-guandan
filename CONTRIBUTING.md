# Contributing

Thanks for your interest in improving guandan-rlcard!

## Getting set up

```bash
git clone https://github.com/Choysang/guandan-rlcard.git
cd guandan-rlcard
pip install -e ".[dev]"
pytest
```

## What we're looking for

* **Rule edge cases** - if the engine generates or accepts an illegal
  combo (or rejects a legal one), open an issue with the hand, the
  table action and the expected result. A failing pytest case is the
  perfect bug report; see `tests/test_action_compare.py` for the style.
* **New baselines** - subclass `guandan_rlcard.agents.GuandanAgent`,
  implement `step` (and optionally `tribute_act`/`back_act`), add an
  entry to `AGENT_REGISTRY` in `guandan_rlcard/baselines/__init__.py`,
  and include a short strategy description in the module docstring.
* **Tensor state encoder** - an optional fixed-size encoding of the
  state dict would make RLCard's built-in RL agents usable here.
* **Performance** - action generation is combinatorial; profiled
  speedups are welcome as long as the action lists stay identical.

## Ground rules

* Keep the engine free of agent/training logic - agents may expose the
  optional `on_episode_end` hook, nothing more.
* Never commit credentials. LLM keys are read from environment
  variables only (`GUANDAN_LLM_API_KEY` etc.).
* Every rule-affecting change needs a regression test, and `pytest`
  must pass.
* Follow PEP 8; docstrings on public functions; English identifiers.
* The rule baselines (`base1`..`base8`) are historical research code -
  bug fixes are welcome, wholesale rewrites should be new baselines
  instead.

## Commit style

```
<type>: <description>
```

with types `feat, fix, refactor, docs, test, chore, perf, ci`.
