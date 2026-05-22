"""
scripts/skill_writer_test.py

Tests for SkillWriter (one-shot skill generator).
Tiers 1-3 use a fake backend — fast, no network.
Tier 4 hits the real API behind --live.

    python scripts/skill_writer_test.py          # offline tests only
    python scripts/skill_writer_test.py --live   # also runs real API test
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.skills.create_skill import (
    SkillWriter,
    StagedSkill,
    SkillValidationError,
)


# ---------- harness ----------

PASS = 0
FAIL = 0

def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

def expect_raises(fn, exc_type, name: str):
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception as e:
        check(name, False, f"raised {type(e).__name__} instead of {exc_type.__name__}: {e}")
    else:
        check(name, False, f"did not raise {exc_type.__name__}")


class FakeBackend:
    """Pre-canned responder. Matches LLMBackend.generate signature."""
    def __init__(self, response: str = ""):
        self.response = response
        self.last_call = None

    def generate(self, system, user, *, temperature=0.7, num_predict=2048,
                 timeout=120.0, stop=None, **opts) -> str:
        self.last_call = {"system": system, "user": user, "opts": opts}
        return self.response


# ---------- canned responses ----------

VALID_RESPONSE = """\
<skill_md>
---
name: test_uptime
description: Returns the macOS uptime.
version: 1.0.0
embodiments: [desktop]
---
Use this when the user asks how long their machine has been running.
</skill_md>
<skill_py>
import subprocess

def handle(**_) -> str:
    try:
        out = subprocess.check_output(["uptime"], text=True, timeout=5)
        return out.strip()
    except Exception as e:
        return f"[error] {e}"
</skill_py>
"""

WITH_BACKTICK_FENCES = """\
<skill_md>
```yaml
---
name: fence_test
description: tests backtick stripping
---
body text
```
</skill_md>
<skill_py>
```python
def handle(**_) -> str:
    return "ok"
```
</skill_py>
"""

WITH_PREAMBLE = """\
Sure! Here's the skill you asked for:

<skill_md>
---
name: preamble_test
description: tests preamble tolerance
---
body
</skill_md>
<skill_py>
def handle(**_) -> str:
    return "ok"
</skill_py>

Let me know if you need any changes!
"""

MISSING_TAGS = "just some text without any tags at all"

BAD_YAML = """\
<skill_md>
---
name: x
description: [unclosed list
---
body
</skill_md>
<skill_py>
def handle(**_) -> str:
    return "ok"
</skill_py>
"""

BAD_NAME = """\
<skill_md>
---
name: 123-not-valid
description: bad name
---
body
</skill_md>
<skill_py>
def handle(**_) -> str:
    return "ok"
</skill_py>
"""

MISSING_DESCRIPTION = """\
<skill_md>
---
name: no_desc
---
body
</skill_md>
<skill_py>
def handle(**_) -> str:
    return "ok"
</skill_py>
"""

PY_SYNTAX_ERROR = """\
<skill_md>
---
name: bad_syntax
description: has syntax error in py
---
body
</skill_md>
<skill_py>
def handle(**_) -> str
    return "missing colon"
</skill_py>
"""

NO_HANDLE = """\
<skill_md>
---
name: no_handle_skill
description: missing handle fn
---
body
</skill_md>
<skill_py>
def other_fn(**_):
    return "ok"
</skill_py>
"""

PARAM_MISMATCH = """\
<skill_md>
---
name: bad_params
description: param mismatch
parameters:
  city:
    type: string
    required: true
    description: city name
---
body
</skill_md>
<skill_py>
def handle(country):
    return country
</skill_py>
"""


# ---------- tests ----------

def test_parsing(tmp_root: Path):
    """Tier 1: just the parser, no validation."""
    print("\n[1/4] parsing")
    md, py = SkillWriter._parse_response(VALID_RESPONSE)
    check("valid response parses", "name: test_uptime" in md and "def handle" in py)

    md, py = SkillWriter._parse_response(WITH_BACKTICK_FENCES)
    check("backtick fences stripped from md", not md.lstrip().startswith("```"))
    check("backtick fences stripped from py", not py.lstrip().startswith("```"))

    md, py = SkillWriter._parse_response(WITH_PREAMBLE)
    check("preamble ignored", "name: preamble_test" in md)

    expect_raises(
        lambda: SkillWriter._parse_response(MISSING_TAGS),
        SkillValidationError,
        "missing tags rejected",
    )

def test_validation(tmp_root: Path):
    """Tier 2: every gate fires the right exception."""
    print("\n[2/4] validation gates")

    def run_with(resp: str):
        return SkillWriter(FakeBackend(resp), staging_root=tmp_root).draft("x")

    expect_raises(lambda: run_with(BAD_YAML), SkillValidationError, "bad YAML rejected")
    expect_raises(lambda: run_with(BAD_NAME), SkillValidationError, "invalid identifier rejected")
    expect_raises(lambda: run_with(MISSING_DESCRIPTION), SkillValidationError, "missing description rejected")
    expect_raises(lambda: run_with(PY_SYNTAX_ERROR), SkillValidationError, "py syntax error rejected")
    expect_raises(lambda: run_with(NO_HANDLE), SkillValidationError, "missing handle() rejected")
    expect_raises(lambda: run_with(PARAM_MISMATCH), SkillValidationError, "param mismatch rejected")


def test_happy_path(tmp_root: Path):
    """Tier 3: full pipeline with fake backend."""
    print("\n[3/4] happy path")
    backend = FakeBackend(VALID_RESPONSE)
    writer = SkillWriter(backend, staging_root=tmp_root)

    staged = writer.draft("an uptime skill")

    check("returns StagedSkill", isinstance(staged, StagedSkill))
    check("name extracted", staged.name == "test_uptime")
    check("staging dir created", staged.staging_dir.is_dir())
    check("SKILL.md written", staged.skill_md_path.is_file())
    check("skill.py written", staged.skill_py_path.is_file())
    check("description passed to backend", backend.last_call["user"] == "an uptime skill")
    check("system prompt passed to backend", "skill writer" in backend.last_call["system"].lower())

    md = staged.skill_md_path.read_text()
    py = staged.skill_py_path.read_text()
    check("SKILL.md has frontmatter", md.count("---") >= 2)
    check("SKILL.md contains name", "test_uptime" in md)
    check("skill.py has handle", "def handle" in py)
    check("skill.py parses", _parses(py))

    # No leftover files for bad attempts
    leftover = list(tmp_root.iterdir())
    check("staging contains only test_uptime", {p.name for p in leftover} == {"test_uptime"})

def test_live_and_execute(tmp_root: Path):
    """Tier 4: real API, generate AND execute each skill."""
    print("\n[4/4] LIVE: generate + execute")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from brain.nlp.backends.anthropic_backend import AnthropicBackend
    from brain.skills.tester import SkillTester

    backend = AnthropicBackend()
    if not backend.api_key:
        check("ANTHROPIC_API_KEY present", False, "skipping live test")
        return

    writer = SkillWriter(backend, staging_root=tmp_root)
    tester = SkillTester()
    prompts = [
        "a skill that returns the current macOS uptime",
        "a skill that lists the names of running applications on macOS",
        "a skill that returns the size of the user's Downloads folder in MB",
    ]
    for p in prompts:
        print(f"\n  prompt: {p}")
        try:
            staged = writer.draft(p)
            check(f"draft '{staged.name}'", True)
        except Exception as e:
            check(f"draft for '{p}'", False, f"{type(e).__name__}: {e}")
            continue

        result = tester.test(staged)
        check(f"execute '{staged.name}'", result.passed, result.error or "")
        if result.output:
            print(f"      output: {result.output[:200]}")
        if not result.passed and result.traceback:
            print(f"      trace: {result.traceback.splitlines()[-1]}")
            
def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also hit the real API")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_parsing(root / "parsing")
        test_validation(root / "validation")
        test_happy_path(root / "happy")

    if args.live:
        with tempfile.TemporaryDirectory() as td:
            test_live_and_execute(Path(td))

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
    
    