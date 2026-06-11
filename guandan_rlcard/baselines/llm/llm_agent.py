"""LLM-driven Guandan agent (template, no keys included).

The agent renders the game state into a Chinese prompt, asks an
OpenAI-compatible model for a legal-action index and falls back to a
random legal action when the reply is missing or unparseable. Tribute
decisions use the rule-following defaults from ``GuandanAgent``.

Usage::

    export GUANDAN_LLM_API_KEY=...      # never commit keys
    export GUANDAN_LLM_BASE_URL=https://api.deepseek.com/
    export GUANDAN_LLM_MODEL=deepseek-chat

    agent = LLMAgent(player_id=1, np_random=np.random.RandomState())
"""

import logging
import re

from guandan_rlcard.agents.base_agent import GuandanAgent
from guandan_rlcard.baselines.llm.client import LLMClient
from guandan_rlcard.baselines.llm.prompts import COMPACT_PROMPT, render_prompt

logger = logging.getLogger('guandan.llm')

_LAST_INT = re.compile(r'-?\d+')


def parse_action_index(reply, num_actions):
    """Extract a legal action index from a model reply.

    Takes the LAST integer in the reply (models often reason first and
    answer last). Returns None when no usable index is found.
    """
    if not reply:
        return None
    matches = _LAST_INT.findall(reply)
    if not matches:
        return None
    index = int(matches[-1])
    if 0 <= index < num_actions:
        return index
    return None


class LLMAgent(GuandanAgent):
    """Agent that delegates action selection to a chat LLM."""

    def __init__(self, player_id, np_random, client=None,
                 template=COMPACT_PROMPT, record_dialogue=False):
        super().__init__(player_id, np_random)
        self.client = client or LLMClient()
        self.template = template
        self.record_dialogue = record_dialogue
        # (prompt, reply, chosen_index) triples when recording is on -
        # handy for building fine-tuning datasets.
        self.dialogue = []
        self.fallback_count = 0

    def step(self, state):
        actions = state['actions']
        if not actions:
            return []
        if len(actions) == 1:
            return actions[0]

        prompt = render_prompt(state, self.template)
        reply = self.client.complete(prompt)
        index = parse_action_index(reply, len(actions))
        if index is None:
            self.fallback_count += 1
            index = self.np_random.randint(len(actions))
            logger.warning('Unparseable LLM reply, falling back to random '
                           '(total fallbacks: %s)', self.fallback_count)
        if self.record_dialogue:
            self.dialogue.append((prompt, reply, index))
        return actions[index]
