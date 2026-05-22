"""
Skill loader — finds skill folders, imports skill.py, validates handler.

A skill folder must contain:
    SKILL.md     — metadata (parsed by spec.py)
    skill.py     — module exporting `def handle(**kwargs) -> str`
"""
from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from brain.skills.spec import SkillSpec, parse_skill_md
from config_metrics.logging import log_debug


@dataclass
class Skill:
    """A loaded skill — metadata + callable handler."""
    spec: SkillSpec
    handler: Callable[..., str]

    @property
    def name(self) -> str:
        return self.spec.name


def _import_skill_module(skill_dir: Path):
    """Import skill.py from the given folder as an isolated module."""
    skill_py = skill_dir / "skill.py"
    if not skill_py.is_file():
        raise FileNotFoundError(f"{skill_dir}: missing skill.py")

    module_name = f"argus_skill_{skill_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, skill_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {skill_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_handler_signature(spec: SkillSpec, handler: Callable) -> None:
    """Ensure handler accepts every required param. Warn on extras."""
    sig = inspect.signature(handler)
    handler_params = sig.parameters

    # Allow **kwargs to satisfy everything
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in handler_params.values()
    )

    for pname, pdef in spec.parameters.items():
        if pdef.required and pname not in handler_params and not has_var_kw:
            raise TypeError(
                f"skill '{spec.name}': handler missing required param '{pname}'"
            )


def load_skill_folder(folder: Path) -> Optional[Skill]:
    """
    Load a single skill folder. Returns None on failure (logs the error)
    so one broken skill doesn't crash startup.
    """
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        return None

    try:
        spec = parse_skill_md(skill_md)
    except ValueError as e:
        log_debug(f"[skills] skipping {folder.name}: {e}")
        return None

    try:
        module = _import_skill_module(folder)
    except Exception as e:
        log_debug(f"[skills] skipping {folder.name}: import failed: {e}")
        return None

    handler = getattr(module, "handle", None)
    if not callable(handler):
        log_debug(f"[skills] skipping {folder.name}: skill.py missing 'handle' function")
        return None

    try:
        _validate_handler_signature(spec, handler)
    except TypeError as e:
        log_debug(f"[skills] skipping {folder.name}: {e}")
        return None

    log_debug(f"[skills] loaded '{spec.name}' from {folder}")
    return Skill(spec=spec, handler=handler)


def discover_skills(root: Path) -> List[Skill]:
    """Scan a directory for skill folders and load each one."""
    if not root.is_dir():
        return []
    skills: List[Skill] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith((".", "_")):
            skill = load_skill_folder(entry)
            if skill is not None:
                skills.append(skill)
    return skills