"""Baseline 4: hand-crafted rule agent (Base4Agent).

Variant of the "rule_parse" family (see base1) with adjusted
decision tables and its own utility helpers.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base4.base4_agent import Base4Agent

__all__ = ['Base4Agent']
