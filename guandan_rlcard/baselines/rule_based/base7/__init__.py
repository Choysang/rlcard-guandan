"""Baseline 7: hand-crafted rule agent (Base7Agent).

Teammate-aware rule agent that returns the chosen action directly
instead of an index. gen_agent.GenAgent is its data-generation
variant used to build LLM fine-tuning datasets.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base7.base7_agent import Base7Agent

__all__ = ['Base7Agent']
