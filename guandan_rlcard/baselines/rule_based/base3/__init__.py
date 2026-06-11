"""Baseline 3: hand-crafted rule agent (Base3Agent).

Rule agent built around point-value estimation helpers
(message_Reyn_CUR): estimates remaining strength before choosing
from the legal action list.

The heuristic core lives in the Action/State classes of this
package, ported verbatim from the original research code.
"""

from guandan_rlcard.baselines.rule_based.base3.base3_agent import Base3Agent

__all__ = ['Base3Agent']
