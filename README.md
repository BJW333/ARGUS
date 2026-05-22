# ARGUS

**Advanced Response and Guidance User System**

A modular, voice-first personal AI assistant built around a platform-agnostic brain, a hybrid local-and-cloud LLM backend, an agentic tool-use loop, a skill-based capability system, project workspace management, a custom-trained wake-word model with production-grade acoustic echo cancellation, and a PySide6/QML liquid-glass interface.

> *"You are ARGUS, an AI assistant built by Blake Weiss. You are calm, confident, and sharp — ditch fluff and filler, get to the point."* — from `config_metrics/argus_system_prompt.txt`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [Brain Layer](#brain-layer)
4. [Embodiment Layer](#embodiment-layer)
5. [Skills + Capabilities](#skills--capabilities)
6. [Workspace System](#workspace-system)
7. [Speech Stack](#speech-stack)
8. [GUI](#gui)
9. [World State](#world-state)
10. [Memory System](#memory-system)
11. [RAG + Web Learning](#rag--web-learning)
12. [Genetic Algorithm Response Optimization](#genetic-algorithm-response-optimization)
13. [Configuration](#configuration)
14. [Data Utilities](#data-utilities)
15. [Legacy Action Handlers](#legacy-action-handlers)
16. [Scripts + Test Harnesses](#scripts--test-harnesses)
17. [Installation](#installation)
18. [Environment Variables](#environment-variables)
19. [Usage](#usage)
20. [Models](#models)
21. [Tech Stack](#tech-stack)
22. [Project Structure](#project-structure)
23. [Status](#status)
24. [Contributing](#contributing)
25. [License](#license)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                              ARGUS                                   │
├──────────────────────────────────────────────────────────────────────┤
│                          Embodiments                                 │
│   embodiments/base.Embodiment       — abstract interface             │
│   embodiments/desktop/embodiment.py — macOS desktop embodiment       │
│   Owns all platform I/O: voice, GUI, OS hooks. Calls brain.process.  │
└──────────────┬───────────────────────────────────────────────────────┘
               │ text in
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              Brain                                   │
│   brain/core.py         — BrainCore: text in, Decision dicts out     │
│   brain/reasoning.py    — LLM pipeline + GA reranking                │
│   brain/planner.py      — multi-step plan decomposition (v1)         │
│   brain/agentic/        — agentic tool-use loop                      │
│   brain/skills/         — skill discovery, registry, generation      │
│   brain/nlp/backends/   — hybrid router: Ollama / Anthropic / OpenAI │
│   brain/memory/         — short/long/personality memory              │
│   brain/rag/            — RAG over scraped knowledge                 │
│   brain/intent.py       — hybrid intent classifier (legacy path)     │
│   brain/reward.py       — response evaluation                        │
└──────────────┬───────────────────────────────────────────────────────┘
               │ Decision dicts
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Skills + Capabilities                            │
│   skills/<name>/SKILL.md + skill.py    — hot-loadable units          │
│   capabilities/registry                — low-level handlers          │
│   services/workspace                   — project workspace manager   │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ read/write
                              ▼
              ┌──────────────────────────────────┐
              │       state/world_state.py       │
              │   WorldState — shared live       │
              │   context. Thread-safe.          │
              │   Callback-driven.               │
              │   Version-tracked.               │
              └──────────────────────────────────┘
```

---

## Design Principles

**1. The brain has no platform imports.** Files inside `brain/` cannot import PySide6, speech libraries, or anything macOS-specific. The brain produces structured `Decision` dicts; embodiments execute them. This is what lets ARGUS swap between desktop, robot, and future bodies without rewriting the reasoning logic.

**2. Shared context lives in one place.** `state.world_state.WORLD` is the single source of truth for live system state. No module hoards private state that other modules need.

**3. Every action is a Capability or a Skill.** Capabilities are low-level handler functions registered in `capabilities/registry.py`. Skills are higher-level, agent-discoverable units (`skills/<name>/SKILL.md + skill.py`) that can wrap capabilities. The agentic loop sees skills as tools.

**4. Hot-loadable everywhere.** Drop a folder into `~/.argus/skills/<name>/` and ARGUS picks it up at startup. User skills override built-ins of the same name.

**5. Hybrid local + cloud LLM with graceful fallback.** Default to local Ollama inference. Fall back to Claude/GPT on weak responses or refusals. Mode is controlled by a single env var.

---

## Brain Layer

### Entry Point — `main_desktop.py`

The desktop entry point. Wires together a `BrainCore` instance, a `DesktopEmbodiment` instance, and the Qt event loop for the QML GUI. Calls `embodiment.run()` which enters the main listen-think-speak loop.

### Core — `brain/core.py`

The single thinking entry point.

- **`BrainCore.process(text: str) -> List[Decision]`** — takes user text in, returns a list of `Decision` dicts of the form `{"type": "action", "capability": "...", "params": {...}}` or `{"type": "response", "text": "...", "confidence": float, "reward": int}`
- Routes between three thinking paths:
  1. **Intent → capability** for simple, recognized intents (the legacy path)
  2. **Agentic tool loop** for compound queries, novel requests, or anything that benefits from multi-tool reasoning
  3. **LLM reasoning pipeline** with GA reranking for free-form responses

No platform imports. Embodiments interpret the returned `Decision` dicts and execute them through their own I/O.

### Reasoning Pipeline — `brain/reasoning.py`

The LLM call orchestrator. Everything else in the brain is routing and bookkeeping; this file does the actual generation work:

- Builds memory-aware prompts (pulls relevant memories from short/long/personality stores)
- Lets the LLM decide whether it needs web knowledge (natural RAG triggering)
- Generates multiple candidate responses with varied temperatures
- Runs GA reranking — crossover, mutation, fitness scoring
- Final low-temperature refinement pass on the winning candidate
- Confidence and reward scoring on the final output

### Planner — `brain/planner.py`

Multi-step plan decomposition for compound requests.

- **v1 (current)** — pass-through. Single intent → single capability → done.
- **v2 (in progress)** — decomposes requests like *"set up my coding environment and show me what I was working on"* into ordered steps:
  1. Start coding workspace
  2. Open IDE
  3. Show recent project notes

The planner is the bridge between simple intent routing and the full agentic loop.

### Intent Classifier — `brain/intent.py` + `brain/nlp/intent.py`

Hybrid intent classifier:
- **Rule-based patterns** for common cases (open/close apps, timers, weather, news, etc.)
- **Fine-tuned ML classifier** for ambiguous cases, hosted on HuggingFace as `bjw333/intent_model_argus`

Returns an intent label that the capability registry maps to a handler.

`brain/intent.py` is a thin wrapper around `brain/nlp/intent.py` for the brain layer — the underlying classifier is platform-agnostic and works for any embodiment.

### Reward System — `brain/reward.py` + `brain/nlp/reward.py`

`DynamicRewardSystem` scores response candidates on:

- **Semantic relevance** — cosine similarity between query and candidate via `sentence-transformers`
- **Sentiment alignment** — VADER sentiment matching
- **Grammar quality** — `LanguageTool` rule violations
- **Clarity and directness** — heuristics for sentence structure and hedging

Used by `reasoning.py` to rank GA candidates, and by `feedback.py` to flag responses for retraining.

### Agentic Tool Loop — `brain/agentic/`

The heart of Phase 2. Replaces brittle JSON-decomposition planning with Claude's native tool-use API.

- **`tool_loop.py`** — `ToolLoop.run()` is the loop engine:
  1. Send message to Claude with the tool list
  2. Claude generates a `tool_use` block
  3. Execute the tool, capture result
  4. Feed result back as `tool_result`
  5. Repeat until Claude emits a final answer (or `max_iterations` hits)

- **`tool_schemas.py`** — `get_tool_definitions()` builds the tool list. Pulls from the skills registry first (`SKILL_REGISTRY`), then falls back to hardcoded tools like `ask_user`. `execute_tool(name, **kwargs)` dispatches calls.

- **`io_hooks.py`** — registry for I/O hooks. Tools that need to talk to the user (e.g., `ask_user`) look up the embodiment's `speak_fn`/`listen_fn` here at execution time. This is the pattern that lets brain-side tools interact with embodiment-side I/O without violating the no-platform-imports rule.

### Skills System — `brain/skills/` + `skills/`

Each skill is a self-contained folder with two files:

```
skills/get_weather/
├── SKILL.md     # YAML frontmatter + markdown body
└── skill.py     # defines handle(**kwargs) -> str
```

**SKILL.md** has YAML frontmatter (`name`, `description`, `version`, `parameters`, `triggers`, `embodiments`) and a markdown body that gives Claude long-form context on when to call this skill.

- **`brain/skills/loader.py`** — low-level skill loading. `discover_skills(root)` walks a directory and returns a list of `Skill` objects; `load_skill_folder(folder)` loads a single skill, validating its SKILL.md and handler signature. Bad SKILL.md files are logged and skipped without crashing.

- **`brain/skills/__init__.py`** — `load_skills()` is the entry point used everywhere else. It calls `discover_skills()` on both `skills/` (built-in) and `~/.argus/skills/` (user) at startup. User folders override built-ins of the same name.

- **`brain/skills/registry.py`** — `SKILL_REGISTRY` holds the active set. `tool_definitions(embodiment=...)` returns the tool list to the agentic loop, filtered by embodiment.

- **`brain/skills/spec.py`** — `parse_skill_md(path)` parses frontmatter into a typed `SkillSpec` dataclass. Validates required fields, parameter manifest, embodiment list.

- **`brain/skills/create_skill.py`** — `SkillWriter`. Given a natural-language description, calls Claude to generate `SKILL.md` + `skill.py`, validates structurally (YAML parses, `handle()` defined, parameters match handler signature), writes to `~/.argus/skills/_staging/<name>/`. Foundation for agent-initiated skill creation.

- **`brain/skills/tester.py`** — `SkillTester`. Dynamically imports a staged skill via `importlib.util`, calls `handle()` with manifest defaults or caller-provided args, captures stdout/stderr/tracebacks, returns a structured `TestResult` with `passed`/`output`/`error`/`traceback`. Used to verify a generated skill works before promotion.

Currently `SkillWriter` and `SkillTester` are developer-facing APIs. Wiring them into the agentic loop as a meta-skill so ARGUS can write its own skills mid-conversation is the next milestone.

### NLP — `brain/nlp/`

- **`chatbot.py`** — `Chatbot` class. Generation orchestration that wires the backend, reward system, and conversation history together. Holds the GA hyperparameters (population size, candidate token cap, mutation rate).
- **`chatbot_init.py`** — `initialize_chatbot_components()`. Bootstraps the chatbot, reward system, and conversation history at startup. Returns the initialized instances to `BrainCore`.

#### NLP Backends — `brain/nlp/backends/`

The hybrid LLM router:

- **`base.py`** — abstract `Backend` interface with `generate(system, user, *, temperature, num_predict, stop, ...)` signature.
- **`ollama_backend.py`** — local inference via Ollama HTTP API. Default model: `argus-40b` (custom modelfile over Qwen2.5-40B-Instruct).
- **`anthropic_backend.py`** — Claude API (default model: `claude-opus-4-7`). Used for the agentic tool loop and harder reasoning.
- **`openai_backend.py`** — OpenAI API fallback.
- **`router.py`** — `RouterBackend`. Three modes via `ARGUS_MODE` env var:
  - `auto` — try primary (API) first, fall back to local on error or refusal
  - `api` — primary only
  - `local` — local only
- **`refusal.py`** — regex-based refusal detector. Scans the head of API responses (~300-500 chars) for policy refusals so the router can fall back to local. Tuned to avoid false positives on legitimate "I can't" responses (e.g., "I can't see the screen right now").

### Memory System — `brain/memory/`

- **`manager.py`** — `MemoryManager` class. Three-tier persistent store backed by JSON files in `core/Argus_memory_storage/`:
  - **`personality.json`** — fixed traits, user preferences, embedding seed
  - **`long_term.json`** — distilled facts and tasks, indexed by sentence embeddings for semantic retrieval
  - **`short_term.json`** — last N conversation turns (configurable, default 10)

Cross-process safety via `filelock`. Semantic search uses `sentence-transformers`. Platform-agnostic — works on desktop or robot via injected callback hooks for prompting and output.

### RAG — `brain/rag/web_rag.py`

Retrieval-augmented generation over a SQLite store of sentence-embedded web content.

- Embeds scraped pages with sentence-transformers
- Stores `(url, title, content, embedding, scraped_at)` rows in SQLite
- Cosine similarity search at query time
- Decay function for old content (configurable freshness)
- Hash-based deduplication

Called from `reasoning.py` when the LLM signals it needs external knowledge.

### Web Learning — `brain/web_learning/`

- **`intelligent_scraper.py`** — async scraper. Quality filtering (skip ads, navigation, boilerplate), rate limiting, domain diversity scoring, async fetch with `aiohttp`, content extraction with `BeautifulSoup`. Selenium fallback for JS-heavy pages.
- **`scraper_service.py`** — background service. Runs the scraper on a cycle (configurable via `ARGUS_WEBLEARN_CYCLE_SEC`). Starts/stops via voice commands (`start web learning`, `stop web learning`). Ingests results into the RAG store.

---

## Embodiment Layer

### Abstract Base — `embodiments/base.py`

`Embodiment(ABC)` — every body must implement:
- `name` property
- `speak(text)` / `listen()` / `display(text)`
- `register_capabilities()` — registers platform-specific capabilities
- `run()` — main listen-think-speak loop

This is what lets the brain stay platform-agnostic. The brain calls these methods; it doesn't know whether `speak()` invokes Mimic3, a robot speaker, or stdout.

### Desktop Embodiment — `embodiments/desktop/embodiment.py`

The macOS implementation. Wraps:
- Mimic3 TTS for speech output
- Google Speech Recognition / Whisper for input
- QML GUI for visual output
- macOS app control (`osascript`, `subprocess`) for opening/closing apps, volume, etc.

Main loop:
1. Listen for wake word (`"Argus"`)
2. Capture user command via speech recognition
3. Pass text to `BrainCore.process()`
4. Execute returned `Decision` dicts (speak, display, run action)

Owns ALL PySide6 and speech imports. The brain never touches any of them.

---

## Skills + Capabilities

### Capabilities — `capabilities/`

- **`registry.py`** — `Capability` dataclass and global `REGISTRY` dict. Every action ARGUS can do is a `Capability` with a name, handler function, and optional `embodiments` restrictions. The brain asks *"what can I do with intent X on embodiment Y?"* and the registry answers.
- **`desktop/`** — macOS-specific capabilities (app control, system info, volume, etc.) registered when the desktop embodiment starts up.
- **`web_learning_cap.py`** — registers `web_learning.start`, `web_learning.stop`, `web_learning.update_knowledge` as capabilities.
- **`workspace_cap.py`** — registers workspace operations (start/resume profile, brain dump, take note, summarize session) as capabilities, bridging the `services/workspace/` module into the registry.

### Built-in Skills — `skills/`

16 skills currently shipped:

| Skill                | Description                                         |
|----------------------|-----------------------------------------------------|
| `get_weather`        | Current weather or forecast for a city              |
| `get_news`           | Latest headlines                                    |
| `get_stock_price`    | Stock quote (uses Alpha Vantage)                    |
| `get_crypto_price`   | Crypto quote                                        |
| `get_flight_status`  | Flight status lookup                                |
| `get_time`           | Current time, optionally for a timezone             |
| `start_timer`        | Set a countdown timer                               |
| `open_app`           | Open a macOS application                            |
| `close_app`          | Close a macOS application                           |
| `set_volume`         | Set system volume                                   |
| `search_wikipedia`   | Search Wikipedia and summarize                      |
| `calculate`          | Evaluate a math expression                          |
| `coin_flip`          | Random heads/tails                                  |
| `tell_joke`          | Tell a joke                                         |
| `check_internet`     | Verify internet connectivity                        |
| `list_capabilities`  | List what ARGUS can currently do                    |

Each is a thin wrapper around `capabilities.REGISTRY` entries. The migration from the old hardcoded `TOOL_DEFINITIONS` was done by `scripts/migrate_tools_to_skills.py`.

---

## Workspace System

`services/workspace/` — voice-driven project workspace manager. See `services/workspace/WORKSPACEREADME.md` for full documentation.

- **`workspace_manager.py`** — `WorkspaceManager` class. Manages workspace sessions with natural voice interaction. Jarvis-style: brief, confident, helpful. Handles profile-based environments (coding, 3D modeling, writing, research, brainstorming).
- **`workspace_integration.py`** — `ArgusWorkspaceIntegration`. The glue between ARGUS and the workspace manager. Takes `speak_fn`, `listen_fn`, `research_fn`, and `sensory_start_fn` callbacks at construction so it stays platform-agnostic. Handles voice commands like *"start my coding workspace"*, *"take a note"*, *"brain dump"*, *"summarize the session"*.
- **`workspace_config.py`** — profile definitions, default project paths, integrations (HELOSFORGE for 3D modeling).
- **`main.py`** — standalone entry point for running the workspace system without the full ARGUS stack.

Features:
- Profile-based workspaces with associated tools/apps
- Voice-triggered start and resume
- Brain dumps — freeform idea capture
- Task tracking
- Auto-generated session summaries
- HELOSFORGE integration for gesture-controlled 3D modeling
- `Sensory_System` integration to disable heavy vision processing when not needed

---

## Speech Stack

### Voice Engine Factory — `speech/voice_engine.py`

Platform-agnostic factory that returns the right AEC engine for the OS:
- macOS → `aec_engine_webrtc`
- Linux → `aec_engine_speex`
- Windows → not yet implemented (WebRTC APM path should work)

Consumers import a single singleton: `from speech.voice_engine import voice_engine as aec_engine`.

Public API: `start()`, `stop()`, `queue_playback()`, `set_audio_callback()`, `is_playing` (property), `clear_playback()`.

### AEC Engines — `speech/aec_engine_webrtc.py`, `speech/aec_engine_speex.py`

Acoustic Echo Cancellation so ARGUS doesn't hear itself talking through the mic.

- **WebRTC APM via LiveKit (macOS)** — production-quality echo cancellation, ~94% echo reduction in testing. The same APM that powers Google Meet and Zoom.
- **Speex (Linux fallback)** — legacy DSP echo cancellation, ~30% reduction. Used where WebRTC APM isn't easily available.

### Voice I/O — `speech/listen.py`, `speech/speak.py`, `speech/speechmanager.py`

- **`listen.py`** — wake-word detection (`"Argus"`) + speech recognition via SpeechRecognition / Whisper. Streams audio through the AEC engine before passing to the recognizer.
- **`speak.py`** — TTS output. Calls Mimic3 for synthesis, routes through the AEC engine's playback queue.
- **`speechmanager.py`** — higher-level voice manager. Pitch and speed control, queued playback, interrupt handling (so ARGUS stops talking mid-sentence when interrupted).

### Wake Word Detection — `models/wakeword/`

Custom-trained model that detects `"Argus"` from a continuous mic stream:
- `argus.onnx` — for ONNX Runtime inference (desktop)
- `argus.tflite` — for TensorFlow Lite (mobile / embedded targets)

Trained on a personal voice dataset; tuned for low false positives in noisy environments.

---

## GUI

`gui_qml/` — PySide6 + QML interface with a liquid-glass aesthetic.

- **`MainView.qml`** — main application window
- **`ChatPanel.qml`** — chat history with animated message bubbles
- **`BrainCanvas.qml`** — 3D audio-reactive neural-network visualization
- **`LiquidGlassCard.qml`** — reusable glass-effect components
- **`FileBrowserPopup.qml`** + **`FolderCard.qml`** — workspace file browsing UI
- **`ArgusTheme.qml`** — shared theme tokens (colors, blur radii, spacing)
- **`backend_bridge.py`** — Python ↔ QML signal bridge. Exposes `BrainCore` outputs and `WorldState` updates to the QML layer; receives user text input from the chat panel and feeds it back to the brain.
- **`core/input_bus.py`** — Qt signal infrastructure for cross-thread GUI updates. `print_to_gui()`, `stream_chunk_to_gui()` so background work can update the UI safely.

---

## World State

`state/world_state.py` — `WORLD` singleton, the shared system context.

```python
from state.world_state import WORLD

WORLD.update("user_input", "what's the weather")
val = WORLD.get("user_input")

# React to changes
def on_change(key, old_val, new_val):
    ...
WORLD.register_callback(on_change)

# Batch updates (single version bump)
with WORLD.batch():
    WORLD.update("active_workspace", "coding")
    WORLD.update("task_state", "executing")
```

Properties:
- **Thread-safe** — internal `RLock` for reads/writes
- **Callback-driven** — modules subscribe to specific keys or all changes
- **Version-tracked** — every change bumps a monotonic version counter; batch updates count as one bump
- **Typed values** — primitives, dicts, lists; not pickled state

Used by the brain (read before thinking), embodiments (write sensor data), services (query for context).

---

## Memory System

`brain/memory/manager.py` — `MemoryManager`.

Three-tier persistent store in `core/Argus_memory_storage/`:

| Tier        | File                | Purpose                                            |
|-------------|---------------------|----------------------------------------------------|
| Personality | `personality.json`  | Fixed traits, user preferences, embedding seed     |
| Long-term   | `long_term.json`    | Distilled facts/tasks, indexed by embeddings       |
| Short-term  | `short_term.json`   | Last N conversation turns (default N=10)           |

- File access is locked via `filelock` for cross-process safety
- Long-term retrieval uses cosine similarity over `sentence-transformers` embeddings
- Short-term decay is FIFO at the configured limit
- New memories are evaluated by the reward system before being persisted (low-quality turns aren't kept)

---

## RAG + Web Learning

### RAG — `brain/rag/web_rag.py`

Retrieval-augmented generation over scraped knowledge.

- SQLite store of `(url, title, content, embedding, scraped_at)` rows
- Cosine similarity over sentence-transformers embeddings
- Configurable freshness decay
- Hash-based deduplication

Triggered from `brain/reasoning.py` when the LLM signals it needs external knowledge — not on every query.

### Intelligent Scraper — `brain/web_learning/intelligent_scraper.py`

Async web scraper with:
- Quality filtering (skip ads, navigation, boilerplate)
- Rate limiting per domain
- Domain diversity scoring (avoid scraping 100 pages from one site)
- Async fetch with `aiohttp`, parsing with `BeautifulSoup`
- Selenium fallback for JS-heavy pages

### Scraper Service — `brain/web_learning/scraper_service.py`

Background service for continuous learning. Runs the scraper on a cycle (default 1 hour, configurable via `ARGUS_WEBLEARN_CYCLE_SEC`). Started/stopped via voice (*"start web learning"*) and ingests results into the RAG store.

---

## Genetic Algorithm Response Optimization

For non-agentic generation, ARGUS refines candidate responses through a GA pass in `brain/reasoning.py`:

1. **Population Generation** — multiple candidate responses generated with varied temperatures
2. **Fitness Evaluation** — each candidate scored by the reward system (semantic relevance + sentiment + grammar + clarity)
3. **Selection** — top performers chosen as parents
4. **Crossover** — semantic-aware text combination of parent responses
5. **Mutation** — semantic drift mutation to explore the response space
6. **Final Refinement** — best candidate passed through a low-temperature generation pass

The GA path is used for free-form questions where there's no clear tool to call. Tool-use questions go through the agentic loop instead, which doesn't need the GA because Claude's responses on a well-defined task are already focused.

---

## Configuration

`config_metrics/`:

- **`main_config.py`** — central config loader. Loads `spaCy en_core_web_md` for NLP, downloads VADER lexicon if missing, exposes `script_dir`, `MASTER` (user's name), and other globals.
- **`logging.py`** — `log_debug()` used throughout the codebase. Writes to `config_metrics/Metrics/chatbot_metrics.log` (gitignored).
- **`argus_system_prompt.txt`** — the system prompt for the LLM. Defines ARGUS's personality (calm, confident, sharp, dry humor) and operational rules.
- **`workspaces.json`** — workspace profile definitions (gitignored — personal). Created at first run.

---

## Data Utilities

`datafunc/`:

- **`data_analysis.py`** — `StockStream` class for the stock price skill. Pulls quotes from Alpha Vantage.
- **`data_store.py`** — generic key-value store with JSON persistence. Used for conversation history, training data exports, and ad-hoc storage by services.

---

## Legacy Action Handlers

`actions/actions.py` — pre-skills action implementations. Includes:

- `calculate(command)` — math expression evaluator
- `timer(userinput)` / `start_timer(userinput)` — timer logic
- `identifynetworkconnect()` — internet check
- `gatherinfofromknowledgebase(query)` — Wikipedia search
- `coin_flip()` — random coin flip
- `cocktail(name)` — cocktail recipe lookup

Being progressively migrated to the skills system. Remains for backward compatibility and as a reference for skills that haven't been migrated yet.

---

## Scripts + Test Harnesses

`scripts/`:

- **`skill_writer_test.py`** — test harness for `SkillWriter` + `SkillTester`. Offline tests (parsing, validation, happy path) plus a `--live` flag that hits the real Anthropic API and generates + executes 3 real skills.
- **`migrate_tools_to_skills.py`** — one-shot migration script. Generates skill folders from the old hardcoded `TOOL_DEFINITIONS`. Safe to re-run; existing skill folders are preserved.
- **`phase2_test.py`** — exercises the agentic tool loop end-to-end with mocked LLM responses.
- **`test_wakeword.py`** — verifies the ONNX/TFLite wake-word models load and run.
- **`test_aec_audio.py`** — LiveKit APM smoke test. Verifies the `AudioProcessingModule` from `livekit.rtc.apm` imports and can process a synthesized frame.
- **`test_compound_classifier.py`** — exercises the intent classifier on multi-intent inputs.
- **`test_mic_perm.py`** — macOS microphone permission check (Sequoia made this more painful).

---

## Installation

### Requirements

- Python 3.10+
- macOS (primary target — adaptable to Linux/Windows with caveats)
- Ollama (for local inference)
- ~8 GB RAM minimum, 16 GB+ recommended for local LLM inference

### Setup

```bash
# Clone
git clone https://github.com/BJW333/ARGUS.git
cd ARGUS

# Dependencies
pip install -r requirements.txt
pip install mycroft-mimic3-tts  # TTS, installed separately

# Pull the local model via Ollama
ollama pull qwen2.5:32b
# OR build the custom modelfile if you have it
ollama create argus-40b -f path/to/Modelfile

# Copy env template and fill in keys
cp .env.example .env
# then edit .env

# Run
python3 main_desktop.py
```

---

## Environment Variables

```bash
# Backend selection
ARGUS_MODE=auto                       # local | api | auto
ARGUS_API_PROVIDER=anthropic          # anthropic | openai
ARGUS_API_MODEL=claude-opus-4-7
ARGUS_LOCAL_MODEL=argus-40b

# API keys
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
ALPHAVANTAGE_API_KEY=                 # stock data skill
OPENWEATHER_API_KEY=                  # weather skill

# Optional overrides
ARGUS_MEMORY_DIR=/path/to/memory      # custom memory location
ARGUS_WEBLEARN_ENABLED=1              # enable background scraping
ARGUS_WEBLEARN_CYCLE_SEC=3600         # scrape cycle in seconds

# Voice / speech tuning (optional)
AEC_INPUT_DEVICE=                     # specific input device for AEC engine
AEC_OUTPUT_DEVICE=                    # specific output device for AEC engine
OWW_MODEL_PATH=                       # override wake-word model path
WAKE_SENSITIVITY=0.5                  # default wake-word threshold
WAKE_SENSITIVITY_SPEAKING=0.35        # threshold while ARGUS is speaking
WHISPER_MODEL=small.en                # Whisper model size for STT
```

---

## Usage

1. Launch: `python3 main_desktop.py`
2. Wait for the greeting
3. Say `"Argus"` to wake, or type into the chat panel
4. Speak or type your query

### Voice Commands

- *"Argus, what's the weather like?"*
- *"Argus, open Spotify"*
- *"Argus, set a timer for 5 minutes"*
- *"Argus, what's the price of Bitcoin?"*
- *"Argus, start my coding workspace"*
- *"Argus, take a note"*
- *"Argus, brain dump"*
- *"Argus, summarize the session"*
- *"Argus, tell me about quantum computing"*

### Special Commands

- `update web knowledge` — ingest scraped content into the RAG store
- `start web learning` / `stop web learning` — toggle background scraping
- `exit` — shutdown ARGUS

---

## Models

- **Local LLM**: `argus-40b` — custom modelfile over Qwen2.5-40B-Instruct, served via Ollama
- **Intent Classification**: [bjw333/intent_model_argus](https://huggingface.co/bjw333/intent_model_argus)
- **Code Generation**: [bjw333/macosargus-code](https://huggingface.co/bjw333/macosargus-code)
- **Object/Face Recognition**: [bjw333/ARGUS_obj_person_recog_model](https://huggingface.co/bjw333/ARGUS_obj_person_recog_model)
- **Wake Word**: custom-trained, `models/wakeword/argus.onnx` and `argus.tflite`

---

## Tech Stack

| Component             | Technology                                    |
|-----------------------|-----------------------------------------------|
| Language              | Python 3.10+, QML                             |
| Local LLM inference   | Ollama                                        |
| API LLM inference     | Anthropic Claude, OpenAI                      |
| Speech recognition    | SpeechRecognition, OpenAI Whisper             |
| Text-to-speech        | Mimic3                                        |
| Acoustic echo cancel  | WebRTC APM via LiveKit (macOS), Speex (Linux) |
| Wake word             | ONNX Runtime, TensorFlow Lite                 |
| NLP                   | spaCy, sentence-transformers, NLTK, VADER     |
| Grammar checking      | LanguageTool                                  |
| ML framework          | PyTorch                                       |
| GUI                   | PySide6 (Qt for Python), QML                  |
| Storage               | SQLite (RAG, web learning)                    |
| File locking          | filelock                                      |
| Web scraping          | aiohttp, BeautifulSoup, Selenium              |

---

## Project Structure

```
ARGUS/
├── main_desktop.py              # Desktop entry point
├── brain/
│   ├── core.py                  # BrainCore — text → Decisions
│   ├── reasoning.py             # LLM pipeline + GA reranking
│   ├── planner.py               # Multi-step plan decomposition (v1)
│   ├── intent.py                # Hybrid intent classifier (brain wrapper)
│   ├── reward.py                # Response evaluation (brain wrapper)
│   ├── agentic/
│   │   ├── tool_loop.py         # ToolLoop.run — the agentic engine
│   │   ├── tool_schemas.py      # Tool definitions + dispatch
│   │   └── io_hooks.py          # I/O hook registry for tools
│   ├── skills/
│   │   ├── loader.py            # Skill discovery
│   │   ├── registry.py          # SKILL_REGISTRY + tool_definitions
│   │   ├── spec.py              # SKILL.md parser
│   │   ├── create_skill.py      # SkillWriter — programmatic generation
│   │   └── tester.py            # SkillTester — staged execution
│   ├── nlp/
│   │   ├── chatbot.py           # Generation orchestration
│   │   ├── chatbot_init.py      # Bootstrap
│   │   ├── intent.py            # Hybrid intent classifier (impl)
│   │   ├── reward.py            # DynamicRewardSystem (impl)
│   │   └── backends/
│   │       ├── base.py          # Abstract Backend interface
│   │       ├── ollama_backend.py
│   │       ├── anthropic_backend.py
│   │       ├── openai_backend.py
│   │       ├── router.py        # RouterBackend (auto/api/local)
│   │       └── refusal.py       # Refusal detector
│   ├── memory/
│   │   └── manager.py           # MemoryManager (three-tier)
│   ├── rag/
│   │   └── web_rag.py           # RAG over scraped content
│   └── web_learning/
│       ├── intelligent_scraper.py
│       └── scraper_service.py
├── skills/                      # Built-in skill folders
│   ├── get_weather/
│   ├── get_news/
│   └── ... (14 more)
├── capabilities/
│   ├── registry.py              # Capability + REGISTRY
│   ├── desktop/                 # macOS-specific capabilities
│   ├── web_learning_cap.py
│   └── workspace_cap.py
├── embodiments/
│   ├── base.py                  # Embodiment ABC
│   ├── desktop/
│   │   └── embodiment.py        # macOS desktop embodiment
│   └── robot/                   # Future
├── speech/
│   ├── voice_engine.py          # Platform-agnostic AEC factory
│   ├── aec_engine_webrtc.py     # WebRTC APM (macOS)
│   ├── aec_engine_speex.py      # Speex fallback (Linux)
│   ├── listen.py                # Wake-word + speech recognition
│   ├── speak.py                 # TTS output
│   └── speechmanager.py         # High-level voice manager
├── state/
│   └── world_state.py           # WORLD — shared system context
├── services/
│   └── workspace/
│       ├── workspace_manager.py
│       ├── workspace_integration.py
│       ├── workspace_config.py
│       ├── main.py              # Standalone entry
│       └── WORKSPACEREADME.md
├── actions/
│   └── actions.py               # Legacy action handlers
├── datafunc/
│   ├── data_analysis.py         # StockStream
│   └── data_store.py            # Generic K/V store
├── core/
│   ├── startup.py               # Greeting / wishme
│   ├── input_bus.py             # Qt signal bridge
│   ├── feedback.py              # Human feedback collection
│   └── Argus_memory_storage/    # Persistent memory (gitignored)
├── gui_qml/
│   ├── MainView.qml
│   ├── ChatPanel.qml
│   ├── BrainCanvas.qml
│   ├── LiquidGlassCard.qml
│   ├── FileBrowserPopup.qml
│   ├── FolderCard.qml
│   ├── ArgusTheme.qml
│   ├── backend_bridge.py        # Python ↔ QML bridge
│   └── qmldir
├── models/
│   └── wakeword/
│       ├── argus.onnx           # ONNX Runtime
│       └── argus.tflite         # TensorFlow Lite
├── scripts/
│   ├── skill_writer_test.py
│   ├── migrate_tools_to_skills.py
│   ├── phase2_test.py
│   ├── test_wakeword.py
│   ├── test_aec_audio.py
│   ├── test_compound_classifier.py
│   └── test_mic_perm.py
├── config_metrics/
│   ├── main_config.py
│   ├── logging.py
│   └── argus_system_prompt.txt
└── data/                        # Training data, audio samples, outputs
```

---

## Status

Active development. Current phase progression:

| Phase | Description                                           | Status      |
|-------|-------------------------------------------------------|-------------|
| 1     | Hybrid backend + performance refactor                 | Complete    |
| 2     | Agentic tool loop                                     | Complete    |
| 3     | Skills system (loader, registry, 16 built-in skills)  | Complete    |
| 4     | Mac-native skills + agent-initiated skill creation    | In progress |
| 5     | MCP integration                                       | Planned     |
| 6     | Telegram embodiment                                   | Planned     |
| 7     | Always-on host                                        | Planned     |
| 8     | Scheduler + proactive skills                          | Planned     |
| 9     | Vision + computer use                                 | Planned     |
| 10    | Robot embodiment                                      | Planned     |

---

## Contributing

Solo project — built and maintained by [@BJW333](https://github.com/BJW333). AI assistance was used for code comments and documentation.

If you find a bug or have suggestions, feel free to open an issue or pull request.

---

## License

MIT
