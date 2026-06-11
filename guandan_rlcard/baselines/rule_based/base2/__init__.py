"""Baseline 2: hand-crafted rule agent (Base2Agent).

Modular rule agent: CreateActionList builds candidate moves,
CountValue scores the hand, and Strategy/PlayCard pick the move
through a staged decision pipeline.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base2.base2_agent import Base2Agent

__all__ = ['Base2Agent']
