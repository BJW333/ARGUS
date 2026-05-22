"""
SKILL.md parser and SkillSpec dataclass.

A SKILL.md file has YAML frontmatter (between --- markers) and a markdown
body. Frontmatter is metadata, body becomes the rich tool description
shown to Claude. The body matters a lot — it's where you tell the model
WHEN to use this skill, including edge cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # requires `pip install pyyaml` if not present


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z",
    re.DOTALL,
)


@dataclass
class SkillParameter:
    """One parameter the skill accepts."""
    type: str                        # "string" | "integer" | "number" | "boolean"
    required: bool = False
    default: Any = None
    description: str = ""


@dataclass
class SkillSpec:
    """
    Parsed metadata from a SKILL.md file.

    Frontmatter fields:
        name              — unique skill id, lowercase_with_underscores
        description       — one-line summary (used for registry + tool short desc)
        version           — semver string, optional
        triggers          — intent labels for the keyword classifier path
        embodiments       — list of embodiment names; [] means works everywhere
        parameters        — map of param_name → SkillParameter
        needs_confirmation — planner will ask before running (future)
    Body (markdown): becomes the long-form tool description for Claude.
    """
    name: str
    description: str
    parameters: Dict[str, SkillParameter] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    embodiments: List[str] = field(default_factory=list)
    version: str = "0.0.0"
    needs_confirmation: bool = False
    body: str = ""                   # markdown body — used as tool description
    source_path: Optional[Path] = None  # set by loader


def parse_skill_md(path: Path) -> SkillSpec:
    """
    Read a SKILL.md file and return a validated SkillSpec.

    Raises ValueError if the file is malformed or missing required fields.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{path}: missing YAML frontmatter (--- delimiters)")

    frontmatter_text, body = match.groups()
    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML frontmatter: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter must be a YAML mapping")

    # Required fields
    for required_key in ("name", "description"):
        if required_key not in data:
            raise ValueError(f"{path}: missing required field '{required_key}'")

    # Build SkillParameter map
    params_raw = data.get("parameters", {}) or {}
    if not isinstance(params_raw, dict):
        raise ValueError(f"{path}: 'parameters' must be a mapping")
    parameters: Dict[str, SkillParameter] = {}
    for pname, pdef in params_raw.items():
        if not isinstance(pdef, dict):
            raise ValueError(f"{path}: parameter '{pname}' must be a mapping")
        parameters[pname] = SkillParameter(
            type=pdef.get("type", "string"),
            required=bool(pdef.get("required", False)),
            default=pdef.get("default"),
            description=pdef.get("description", ""),
        )

    return SkillSpec(
        name=data["name"],
        description=data["description"],
        parameters=parameters,
        triggers=list(data.get("triggers", []) or []),
        embodiments=list(data.get("embodiments", []) or []),
        version=str(data.get("version", "0.0.0")),
        needs_confirmation=bool(data.get("needs_confirmation", False)),
        body=body.strip(),
        source_path=path,
    )