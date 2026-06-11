"""Baseline 8: hand-crafted rule agent (Base8Agent).

Rule agent of the parse-index family with its own State encoder
and decision tables.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base8.base8_agent import Base8Agent

__all__ = ['Base8Agent']
