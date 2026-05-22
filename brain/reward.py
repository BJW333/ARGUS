"""
ARGUS Brain Reward System
=========================

Re-exports the existing DynamicRewardSystem from nlp/reward.py.
That module is already platform-agnostic — no edits needed.
"""
from __future__ import annotations

from brain.nlp.reward import DynamicRewardSystem

__all__ = ["DynamicRewardSystem"]
