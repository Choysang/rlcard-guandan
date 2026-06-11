"""Baseline 1: hand-crafted rule agent (Base1Agent).

Rule agent of the "rule_parse" family: decomposes the hand into
combos and walks fixed priority tables, informed by remaining-card
counting (remain_cards) and the pass counters of both sides.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base1.base1_agent import Base1Agent

__all__ = ['Base1Agent']
