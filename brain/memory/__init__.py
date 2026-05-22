"""
Brain Memory subsystem.

Currently uses core.memory_system.MemoryManager directly.
Future: LLM-based fact extraction, topic clusters, episodic memory.

The memory_system.py import path stays the same for now — BrainCore
imports it in __init__. This package exists so we can migrate
memory code here incrementally without breaking anything.
"""
