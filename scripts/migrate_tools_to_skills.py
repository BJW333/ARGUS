"""
scripts/migrate_tools_to_skills.py

Auto-convert TOOL_DEFINITIONS entries from brain/agentic/tool_schemas.py
into skills/<name>/SKILL.md + skill.py folders.

Run from the project root:
    python scripts/migrate_tools_to_skills.py

Behavior:
    - Idempotent — won't overwrite existing skill folders.
    - Reads TOOL_DEFINITIONS at import time, so any new tool you add to
      tool_schemas will be picked up next run.
    - Preserves prompt engineering: full tool description → SKILL.md body,
      input_schema param descriptions → frontmatter verbatim.

To customize:
    - MIGRATION_CONFIG: maps tool name → wrapped capability + call mode
    - RENAME_MAP: optional, output skill under a different folder name
    - SKIP_LIST: tools to deliberately not migrate
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from brain.agentic.tool_schemas import TOOL_DEFINITIONS  # noqa: E402

SKILLS_DIR = PROJECT_ROOT / "skills"


# ─────────────────────────────────────────────────────────────────────
# Config — edit these three blocks to control what gets generated
# ─────────────────────────────────────────────────────────────────────

# Maps tool name → how to wrap it.
#   mode="passthrough": forward kwargs 1:1 to cap.handler
#   mode="no_args":     cap.handler() with no kwargs
#   mode="custom":      paste `body` directly as the function body
MIGRATION_CONFIG = {
    "get_weather":       {"wraps": "weather.get_for_city",   "mode": "passthrough"},
    "get_stock_price":   {"wraps": "stocks.get_for_company", "mode": "passthrough"},
    "get_crypto_price":  {"wraps": "crypto.get_for_coin",    "mode": "passthrough"},
    "get_time":          {"wraps": "time.get_current",       "mode": "no_args"},
    "get_news":          {"wraps": "news.get_headlines",     "mode": "no_args"},
    "check_internet":    {"wraps": "network.check",          "mode": "no_args"},
    "calculate":         {"wraps": "calculator.calculate",   "mode": "passthrough"},
    "get_flight_status": {"wraps": "flights.get_for_number", "mode": "passthrough"},
    "coin_flip":         {"wraps": "misc.coinflip",          "mode": "no_args"},
    "tell_joke":         {"wraps": "misc.joke",              "mode": "no_args"},
    "list_capabilities": {"wraps": "misc.list_skills",       "mode": "no_args"},

    # Tools that translate args before calling the capability
    "open_app": {
        "wraps": "desktop.open_app",
        "mode": "custom",
        "body": '    return cap.handler(raw_text=f"open {app_name}", followup_answer=app_name)',
    },
    "close_app": {
        "wraps": "desktop.close_app",
        "mode": "custom",
        "body": '    return cap.handler(raw_text=f"close {app_name}", followup_answer=app_name)',
    },
    "set_volume": {
        "wraps": "desktop.volume",
        "mode": "custom",
        "body": '    return cap.handler(raw_text=command)',
    },
    "start_timer": {
        "wraps": "timer.start",
        "mode": "custom",
        "body": '    return cap.handler(raw_text=command)',
    },
    "search_wikipedia": {
        "wraps": "search.wikipedia",
        "mode": "custom",
        "body": '    return cap.handler(raw_text=query, entities={query: "MISC"})',
    },
}

# Optional — rename a tool's output folder to something different.
# Uncomment "get_weather": "weather" if you want to merge the auto-generated
# version with your manually-migrated skills/weather/ folder (it'll be
# detected as "already exists" and skipped).
RENAME_MAP = {
    # "get_weather": "weather",
}

# Tools to deliberately not migrate.
SKIP_LIST = {
    "ask_user": "I/O hook (lives in brain/agentic/io_hooks.py), not a capability",
}


# ─────────────────────────────────────────────────────────────────────
# Template rendering
# ─────────────────────────────────────────────────────────────────────

_PYTYPE = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}


def render_skill_md(tool: dict) -> str:
    """Build SKILL.md content from a tool definition."""
    name = tool["name"]
    description = tool["description"]
    schema = tool.get("input_schema", {}) or {}
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    # First sentence becomes the YAML `description` field; full text → body
    short = description.split(". ")[0].rstrip(".") + "."

    lines = [
        "---",
        f"name: {name}",
        f"description: {short}",
        "version: 1.0.0",
        "triggers: []",
        "embodiments: []",
    ]

    if properties:
        lines.append("parameters:")
        for pname, pdef in properties.items():
            lines.append(f"  {pname}:")
            lines.append(f"    type: {pdef.get('type', 'string')}")
            lines.append(f"    required: {str(pname in required).lower()}")
            if "default" in pdef:
                lines.append(f"    default: {pdef['default']}")
            desc = pdef.get("description", "").replace("\n", " ").replace('"', '\\"')
            if desc:
                lines.append(f'    description: "{desc}"')
    else:
        lines.append("parameters: {}")

    lines.append("---")
    lines.append("")
    lines.append(description)
    lines.append("")

    return "\n".join(lines)


def render_skill_py(tool: dict, config: dict) -> str:
    """Build skill.py content for a tool."""
    name = tool["name"]
    wraps = config["wraps"]
    mode = config["mode"]
    schema = tool.get("input_schema", {}) or {}
    properties = schema.get("properties", {}) or {}

    # Build function signature from properties
    sig_parts = []
    for pname, pdef in properties.items():
        ptype = _PYTYPE.get(pdef.get("type", "string"), "str")
        if "default" in pdef:
            sig_parts.append(f"{pname}: {ptype} = {pdef['default']!r}")
        else:
            sig_parts.append(f"{pname}: {ptype}")
    sig_parts.append("**_")
    sig = ", ".join(sig_parts)

    # Build the call body
    if mode == "passthrough":
        kwargs = ", ".join(f"{p}={p}" for p in properties.keys())
        body = f"    return cap.handler({kwargs})"
    elif mode == "no_args":
        body = "    return cap.handler()"
    elif mode == "custom":
        body = config["body"]
    else:
        body = '    raise NotImplementedError("unknown mode")'

    return f'''"""
{name} skill — auto-generated from TOOL_DEFINITIONS.

Wraps the existing {wraps} capability. Behavior is unchanged from the
pre-migration hardcoded handler in brain/agentic/tool_schemas.py.

Generated by scripts/migrate_tools_to_skills.py — feel free to edit by hand
once it's in your repo. Re-running the migration script will skip this file
because the folder already exists.
"""
from __future__ import annotations


def handle({sig}) -> str:
    from capabilities.registry import REGISTRY
    cap = REGISTRY.get("{wraps}")
    if cap is None:
        return "[error] {wraps} not registered"
{body}
'''


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SKILLS_DIR.mkdir(exist_ok=True)

    generated, exists, skipped, unhandled = [], [], [], []

    for tool in TOOL_DEFINITIONS:
        name = tool["name"]

        if name in SKIP_LIST:
            skipped.append((name, SKIP_LIST[name]))
            continue

        if name not in MIGRATION_CONFIG:
            unhandled.append(name)
            continue

        target = RENAME_MAP.get(name, name)
        folder = SKILLS_DIR / target

        if folder.exists():
            exists.append(target)
            continue

        folder.mkdir()
        (folder / "SKILL.md").write_text(render_skill_md(tool))
        (folder / "skill.py").write_text(
            render_skill_py(tool, MIGRATION_CONFIG[name])
        )
        generated.append(target)

    # ── summary ──
    sep = "─" * 60
    print(f"\n{sep}\nMigration summary\n{sep}")

    print(f"\nGenerated ({len(generated)}):")
    for n in generated:
        print(f"  ✓ skills/{n}/")

    if exists:
        print(f"\nAlready existed, skipped ({len(exists)}):")
        for n in exists:
            print(f"  - skills/{n}/")

    if skipped:
        print(f"\nSkipped by SKIP_LIST ({len(skipped)}):")
        for n, reason in skipped:
            print(f"  - {n}: {reason}")

    if unhandled:
        print(f"\n⚠  UNHANDLED — no MIGRATION_CONFIG entry ({len(unhandled)}):")
        for n in unhandled:
            print(f"  ! {n}: add an entry and re-run")

    print()


if __name__ == "__main__":
    main()
