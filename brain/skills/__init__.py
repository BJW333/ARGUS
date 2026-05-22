"""
ARGUS skills system — hot-loadable capability folders.

Public API:
    from brain.skills import load_skills, SKILL_REGISTRY
    load_skills()                                     # finds builtin + user skills
    SKILL_REGISTRY.tool_definitions(embodiment="desktop")
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from brain.skills.loader import Skill, discover_skills
from brain.skills.registry import SKILL_REGISTRY
from config_metrics.logging import log_debug


# Resolve the builtin skills dir relative to the project root.
# (this file lives at brain/skills/__init__.py — go up 2 levels)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILTIN_SKILLS_DIR = _PROJECT_ROOT / "skills"
USER_SKILLS_DIR = Path.home() / ".argus" / "skills"


def load_skills() -> List[Skill]:
    """
    Discover and register all builtin + user skills.
    Returns the list of loaded skills (also accessible via SKILL_REGISTRY).

    User skills are loaded AFTER builtin so they can override by name.
    """
    USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    loaded: List[Skill] = []
    for source in (BUILTIN_SKILLS_DIR, USER_SKILLS_DIR):
        skills = discover_skills(source)
        for skill in skills:
            SKILL_REGISTRY.add(skill)
            loaded.append(skill)

    log_debug(f"[skills] loaded {len(loaded)} skills "
              f"(builtin={BUILTIN_SKILLS_DIR}, user={USER_SKILLS_DIR})")
    return loaded


__all__ = ["load_skills", "SKILL_REGISTRY", "Skill"]