"""Baseline 5: hand-crafted rule agent (Base5Agent).

Variant of the "rule_parse" family (see base1); served as the
standard opponent in most of the original experiments.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base5.base5_agent import Base5Agent

__all__ = ['Base5Agent']
