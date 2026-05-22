"""
SkillRegistry — central index of loaded skills + bridge to existing layers.

A skill, once loaded, gets surfaced two places:
    1. capabilities.REGISTRY  — so keyword-routed intent classification still works.
    2. brain.agentic.tool_schemas — so the agentic loop can call it as a tool.

Both bridges are idempotent (re-registering replaces, doesn't duplicate).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.skills.loader import Skill
from capabilities.registry import REGISTRY, Capability
from config_metrics.logging import log_debug


class SkillRegistry:
    """Holds loaded skills; exposes them to the two downstream systems."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        """Register a loaded skill into the global capability + tool layers."""
        # Overwrite if already present (later loads win — useful for user skills
        # overriding builtins)
        self._skills[skill.name] = skill

        # ── 1. Bridge to capabilities.REGISTRY ──
        REGISTRY.register(Capability(
            name=f"skill.{skill.name}",
            description=skill.spec.description,
            handler=skill.handler,
            keywords=skill.spec.triggers,
            required_embodiments=skill.spec.embodiments,
            parameters={p: pdef.type for p, pdef in skill.spec.parameters.items()},
            needs_confirmation=skill.spec.needs_confirmation,
        ))

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all(self) -> List[Skill]:
        return list(self._skills.values())

    def tool_definitions(self, embodiment: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Produce Anthropic-format tool schemas for the agentic loop.

        If `embodiment` is given, filters out skills that don't support it.
        """
        out = []
        for skill in self._skills.values():
            if (skill.spec.embodiments
                    and embodiment
                    and embodiment not in skill.spec.embodiments):
                continue

            properties: Dict[str, Any] = {}
            required: List[str] = []
            for pname, pdef in skill.spec.parameters.items():
                properties[pname] = {
                    "type": pdef.type,
                    "description": pdef.description,
                }
                if pdef.default is not None:
                    properties[pname]["default"] = pdef.default
                if pdef.required:
                    required.append(pname)

            description = skill.spec.description
            if skill.spec.body:
                description = f"{description}\n\n{skill.spec.body}"

            out.append({
                "name": skill.name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return out

    def execute(self, name: str, **kwargs) -> str:
        """Run a skill by name. Returns string for the tool_result block."""
        skill = self._skills.get(name)
        if skill is None:
            return f"[error] unknown skill: {name!r}"
        try:
            result = skill.handler(**kwargs)
        except TypeError as e:
            return f"[error] bad arguments for {name}: {e}"
        except Exception as e:
            log_debug(f"[skill {name}] exception: {e}")
            return f"[error] {name} failed: {e}"
        if result is None:
            return "Done."
        return str(result)


SKILL_REGISTRY = SkillRegistry()