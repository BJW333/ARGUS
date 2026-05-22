#!/usr/bin/env python3
"""
Standalone smoke test for the argus-router compound classifier.

Tests the exact user_prompt that Planner._looks_compound will send,
without requiring the full ARGUS process to be running.

Usage:
    python3 test_compound_classifier.py

Requires: Ollama running locally with argus-router model installed.
"""
import sys
import time
import requests


def make_prompt(text: str) -> str:
    """Build the exact user_prompt that planner.py will send."""
    return (
        'You are a binary classifier. Output exactly one word: "single" or "multiple". '
        'Lowercase. No punctuation. No explanation. Never attempt or answer the request.\n'
        '\n'
        'Definitions:\n'
        '- single = one task, even if it has multiple requirements joined by "and"\n'
        '- multiple = two or more independent tasks executed separately\n'
        '\n'
        'Examples:\n'
        '\n'
        'User: open Discord\n'
        'Answer: single\n'
        '\n'
        'User: write a program that sorts a list and uses recursion\n'
        'Answer: single\n'
        '\n'
        'User: build a trading bot that uses RSI and MACD\n'
        'Answer: single\n'
        '\n'
        'User: explain how TCP works and why it matters\n'
        'Answer: single\n'
        '\n'
        'User: find me a laptop that is lightweight and has good battery\n'
        'Answer: single\n'
        '\n'
        'User: open Spotify and tell me the weather\n'
        'Answer: multiple\n'
        '\n'
        'User: set a timer for 5 minutes then send a message to John\n'
        'Answer: multiple\n'
        '\n'
        'User: close Slack and open Notion\n'
        'Answer: multiple\n'
        '\n'
        'User: check the weather and play music\n'
        'Answer: multiple\n'
        '\n'
        f'User: {text}\n'
        'Answer:'
    )


def classify(text: str) -> tuple[str, float]:
    """Call argus-router; return (raw_answer, elapsed_seconds)."""
    t0 = time.perf_counter()
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "argus-router-v2",
            "stream": False,
            "messages": [{"role": "user", "content": make_prompt(text)}],
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": 3},
        },
        timeout=15,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - t0
    raw = (response.json().get("message", {}).get("content", "") or "").strip().lower()
    return raw, elapsed


# (input, expected_label, category)
CASES = [
    # ── Clear single-task cases ──
    ("open Spotify",                                                "single",   "trivial single"),
    ("what time is it",                                             "single",   "trivial single"),
    ("explain how TCP works",                                       "single",   "trivial single"),

    # ── The "and"-trap: single tasks that contain "and" ──
    ("write a program that uses recursion and sorts a list",        "single",   "and-trap"),
    ("build a trading bot that uses RSI and MACD",                  "single",   "and-trap"),
    ("find me a laptop that is lightweight and has good battery",   "single",   "and-trap"),
    ("explain how TCP works and why it matters",                    "single",   "and-trap"),
    ("write a Python script that reads a CSV and prints totals",    "single",   "and-trap"),

    # ── Clear compound cases ──
    ("open Spotify and tell me the weather",                        "multiple", "the failing test"),
    ("open Spotify and tell me the weather in Red Bank",            "multiple", "the failing test"),
    ("close Slack and open Notion",                                 "multiple", "clear compound"),
    ("check the weather and play music",                            "multiple", "clear compound"),
    ("open Chrome and set a timer for 10 minutes",                  "multiple", "clear compound"),

    # ── Sequential signal cases (would shortcut to True before LLM in real planner) ──
    ("set a timer for 5 minutes then send a message to John",       "multiple", "then-signal"),
    ("close Spotify then open VS Code",                             "multiple", "then-signal"),
]


def main():
    print("Testing argus-router compound classifier")
    print("=" * 78)
    print(f"{'res':5} {'expected':10} {'got':10} {'time':>7}  input")
    print("-" * 78)

    passed = 0
    failed = 0
    times = []

    for text, expected, category in CASES:
        try:
            raw, elapsed = classify(text)
            times.append(elapsed)
            actual = "multiple" if raw.startswith("multi") else "single"
            ok = actual == expected
            mark = "PASS" if ok else "FAIL"
            print(f"{mark:5} {expected:10} {actual:10} {elapsed:6.2f}s  {text}")
            if not ok:
                print(f"      └─ raw={raw!r}   category={category}")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR {expected:10} {'—':10} {'—':>7}  {text}")
            print(f"      └─ {e}")
            failed += 1

    print("-" * 78)
    avg = sum(times) / len(times) if times else 0
    print(f"{passed}/{passed + failed} passed  |  avg latency: {avg:.2f}s")

    if failed == 0:
        print("\nClassifier is clean. Safe to wire up _looks_compound in the live system.")
    else:
        print(f"\n{failed} failing case(s) — see raw outputs above before wiring up.")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
