"""
brain/skills/writer.py

One-shot skill writer. Step 1 of the recursive skill-generation system.

Given a natural-language description, calls Claude via the Anthropic backend
to generate a SKILL.md + skill.py pair, validates the output, and writes
both files to ~/.argus/skills/_staging/<name>/.

Does NOT execute the skill. Does NOT promote to live. Those are later steps.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from config_metrics.logging import log_debug


SKILL_WRITER_SYSTEM_PROMPT = """\
You are a skill writer for the ARGUS agent system. Given a natural-language
description of a desired capability, you produce two files that together
define an ARGUS skill.

# Output format

Respond with both files wrapped in XML tags. No commentary before, between,
or after the tags. Exactly this shape:

<skill_md>
[contents of SKILL.md here]
</skill_md>
<skill_py>
[contents of skill.py here]
</skill_py>

# SKILL.md format

YAML frontmatter between `---` delimiters, then a markdown body.

Frontmatter fields:
- name: required. snake_case Python identifier (e.g. `battery_status`).
- description: required. One sentence explaining what the skill does.
- version: optional. Semver string. Default "1.0.0".
- parameters: optional. Mapping of name -> {type, required, default, description}.
- embodiments: optional. List of embodiments where this skill is available
  (e.g. ["desktop"]). Omit to make the skill available everywhere.

The body (everything after the closing `---`) is the long-form description
Claude reads at runtime when deciding whether to invoke the skill. Write
2-4 sentences describing exactly when this skill should be used.

# skill.py format

Must define a top-level function `handle(**kwargs) -> str`. Parameters
declared in the manifest arrive as keyword arguments. Return a string
that gets shown to the user.

Wrap the body in try/except and return errors as `[error] <message>`
strings rather than raising. This lets the agentic loop recover gracefully.

# Example

<skill_md>
---
name: battery_status
description: Returns the current battery percentage and charging state on macOS.
version: 1.0.0
embodiments: [desktop]
---
Use this skill when the user asks about their laptop battery, power level,
or whether the machine is plugged in. Returns a short status string like
"Battery at 72%, charging".
</skill_md>
<skill_py>
import subprocess
import re

def handle(**_) -> str:
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "batt"], text=True, timeout=5
        )
        m = re.search(r"(\d+)%;\s*(\S+)", out)
        if not m:
            return "[error] could not parse pmset output"
        pct, state = m.group(1), m.group(2).replace("_", " ")
        return f"Battery at {pct}%, {state}"
    except Exception as e:
        return f"[error] battery_status failed: {e}"
</skill_py>

Follow that exact shape for every skill you generate.
"""


@dataclass
class StagedSkill:
    """Result of a successful draft. Caller uses this to find the files."""
    name: str
    staging_dir: Path
    skill_md_path: Path
    skill_py_path: Path


class SkillValidationError(Exception):
    """Raised when the generated skill fails structural validation."""


class SkillWriter:
    """
    One-shot skill writer. Stateless across calls.

    Usage:
        from brain.nlp.backends.anthropic_backend import AnthropicBackend
        writer = SkillWriter(AnthropicBackend())
        staged = writer.draft("a skill that returns my battery percentage")
        print(staged.skill_md_path.read_text())
    """

    def __init__(
        self,
        backend,
        staging_root: Optional[Path] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ):
        self.backend = backend
        self.staging_root = staging_root or (
            Path.home() / ".argus" / "skills" / "_staging"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---------- public ----------

    def draft(self, description: str) -> StagedSkill:
        """Generate a skill from a description and stage it. Raises on failure."""
        raw = self._generate(description)
        skill_md, skill_py = self._parse_response(raw)
        name = self._validate(skill_md, skill_py)
        return self._write_to_staging(name, skill_md, skill_py)

    # ---------- private ----------

    def _generate(self, description: str) -> str:
        return self.backend.generate(
            SKILL_WRITER_SYSTEM_PROMPT,   # positional
            description,                   # positional
            temperature=self.temperature,
            num_predict=self.max_tokens,
            stop=[],
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[str, str]:
        """Extract <skill_md>...</skill_md> and <skill_py>...</skill_py>."""
        md = re.search(r"<skill_md>\s*(.*?)\s*</skill_md>", text, re.DOTALL)
        py = re.search(r"<skill_py>\s*(.*?)\s*</skill_py>", text, re.DOTALL)
        if not md or not py:
            raise SkillValidationError(
                "response missing <skill_md> or <skill_py> tags"
            )
        return SkillWriter._clean(md.group(1)), SkillWriter._clean(py.group(1))

    @staticmethod
    def _clean(text: str) -> str:
        """Strip whitespace and ```language fences if Claude added any."""
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        return text.strip()

    def _validate(self, skill_md: str, skill_py: str) -> str:
        """Run gates. Return skill name on success, raise on failure."""
        # Frontmatter delimiters present
        if skill_md.count("---") < 2:
            raise SkillValidationError("SKILL.md missing frontmatter delimiters")

        # YAML parses
        try:
            _, fm, _body = skill_md.split("---", 2)
            meta = yaml.safe_load(fm) or {}
        except (ValueError, yaml.YAMLError) as e:
            raise SkillValidationError(f"SKILL.md YAML invalid: {e}")

        # name field is a valid identifier
        name = meta.get("name")
        if not name or not str(name).isidentifier():
            raise SkillValidationError(
                f"SKILL.md 'name' missing or not a valid identifier: {name!r}"
            )

        # description present
        if not meta.get("description"):
            raise SkillValidationError("SKILL.md 'description' missing or empty")

        # collision against live skills (not staging — restaging is fine)
        live_dir = Path.home() / ".argus" / "skills" / name
        if live_dir.exists():
            raise SkillValidationError(
                f"skill '{name}' already exists at {live_dir}"
            )

        # skill.py parses as valid Python
        try:
            tree = ast.parse(skill_py)
        except SyntaxError as e:
            raise SkillValidationError(f"skill.py syntax error: {e}")

        # skill.py defines handle()
        handle_fn = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "handle"),
            None,
        )
        if handle_fn is None:
            raise SkillValidationError(
                "skill.py missing top-level handle() function"
            )

        # every required manifest param is in handle()'s signature (unless **kwargs)
        sig_args = {a.arg for a in handle_fn.args.args}
        has_kwargs = handle_fn.args.kwarg is not None
        for pname, pdef in (meta.get("parameters") or {}).items():
            required = isinstance(pdef, dict) and pdef.get("required")
            if required and pname not in sig_args and not has_kwargs:
                raise SkillValidationError(
                    f"manifest declares required param {pname!r} "
                    f"but handle() doesn't accept it"
                )

        return name

    def _write_to_staging(
        self, name: str, skill_md: str, skill_py: str
    ) -> StagedSkill:
        staging_dir = self.staging_root / name
        staging_dir.mkdir(parents=True, exist_ok=True)

        md_path = staging_dir / "SKILL.md"
        py_path = staging_dir / "skill.py"
        md_path.write_text(skill_md)
        py_path.write_text(skill_py)

        log_debug(f"[skill_writer] staged '{name}' at {staging_dir}")
        return StagedSkill(
            name=name,
            staging_dir=staging_dir,
            skill_md_path=md_path,
            skill_py_path=py_path,
        )