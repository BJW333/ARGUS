# brain/skills/tester.py
"""
brain/skills/tester.py

Executes staged skills in isolation. Step 2 of the recursive skill-generation
system. Imports skill.py dynamically, calls handle() with manifest defaults
(or caller-provided args), captures output/errors, returns a structured result.

Does NOT sandbox write operations — that's a step-3 concern.
Does NOT promote skills to live — separate component.
"""
from __future__ import annotations

import importlib.util
import io
import traceback as tb_module
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from brain.skills.create_skill import StagedSkill


@dataclass
class TestResult:
    passed: bool
    output: str
    error: Optional[str] = None
    traceback: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class SkillExecutionError(Exception):
    """Raised when a skill can't be loaded — separate from runtime errors."""


class SkillTester:
    """
    Runs staged skills in the current process. Stateless across calls.

    Usage:
        tester = SkillTester()
        result = tester.test(staged_skill)
        if result.passed:
            print(result.output)
        else:
            print(result.error or result.output)
    """

    def test(
        self, staged: StagedSkill, args: Optional[dict] = None
    ) -> TestResult:
        # --- import phase (structural failures) ---
        try:
            module = self._import_skill_module(
                staged.skill_py_path,
                f"argus_skill_{staged.name}",
            )
        except SkillExecutionError as e:
            return TestResult(
                passed=False,
                output="",
                error=str(e),
                traceback=tb_module.format_exc(),
            )
        except Exception as e:
            # skill.py had import-time errors (e.g. bad imports, syntax slipped past ast)
            return TestResult(
                passed=False,
                output="",
                error=f"import failed: {e}",
                traceback=tb_module.format_exc(),
            )

        if not hasattr(module, "handle"):
            return TestResult(
                passed=False,
                output="",
                error="skill.py must define a handle(**kwargs) function",
            )
        handle = module.handle

        # --- argument resolution ---
        if args is None:
            args = self._defaults_from_manifest(staged.skill_md_path)

        # --- execution phase (runtime failures) ---
        out_buf, err_buf = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                result = handle(**args)
        except Exception as e:
            return TestResult(
                passed=False,
                output="",
                error=str(e),
                traceback=tb_module.format_exc(),
                stdout=out_buf.getvalue(),
                stderr=err_buf.getvalue(),
            )

        # --- result interpretation ---
        # Skills signal failure via "[error] ..." prefix instead of raising.
        output_str = str(result) if result is not None else ""
        passed = not output_str.startswith("[error]")

        return TestResult(
            passed=passed,
            output=output_str,
            error=output_str if not passed else None,
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
        )

    # ---------- helpers ----------

    def _import_skill_module(self, path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise SkillExecutionError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _defaults_from_manifest(self, md_path: Path) -> dict:
        text = md_path.read_text()
        try:
            _, fm, _ = text.split("---", 2)
        except ValueError:
            return {}
        meta = yaml.safe_load(fm) or {}
        params = meta.get("parameters") or {}
        return {
            name: pdef["default"]
            for name, pdef in params.items()
            if isinstance(pdef, dict) and "default" in pdef
        }