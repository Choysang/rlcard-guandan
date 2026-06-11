"""Baseline 6: hand-crafted rule agent (Base6Agent).

Rule agent with dedicated endgame analysis (lasthand): switches
to exhaustive last-hand reasoning when hands get short.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base6.base6_agent import Base6Agent

__all__ = ['Base6Agent']
