# ARGUS Workspace System

Voice-driven project workspace manager for ARGUS. Automatically launches the right tools for different types of work, manages project notes, and integrates with Sensory_System for power management.

## Features

- **Profile-based workspaces** - Coding, 3D Modeling, Writing, Research, Brainstorming
- **Voice-driven** - "Argus, start my 3D modeling workspace"
- **HELOSFORGE integration** - Launches your gesture-controlled 3D modeler
- **Project organization** - Auto-creates project folders and session notes
- **Brain dumps** - Capture freeform ideas via voice
- **Task tracking** - Add tasks to your project
- **Session summaries** - Auto-summarize your work at session end
- **Sensory_System integration** - Disables heavy vision processing when not needed

## Quick Start

### Standalone Usage

```bash
# Interactive profile selection
python main.py

# Direct profile start
python main.py --profile 3d_modeling

# Resume last workspace
python main.py --resume

# List available profiles
python main.py --list-profiles
```

### ARGUS Integration

```python
# In your ARGUS main.py
from workspace_integration import ArgusWorkspaceIntegration

# Initialize at startup
workspace = ArgusWorkspaceIntegration(
    speak_fn=speak,
    listen_fn=generalvoiceinput,
    sensory_hub=sensory_hub,  # Optional
)

# In your command handler (after wake word)
def handle_command(command):
    # Check workspace commands first
    if workspace.handle_command(command):
        return
    
    # ... rest of your ARGUS commands
```

## Voice Commands

### Starting Workspaces

| Say this | What happens |
|----------|--------------|
| "Start my workspace" | Interactive profile selection |
| "Start coding workspace" | Opens VS Code, Terminal, Notion |
| "Start 3D modeling workspace" | Launches HELOSFORGE, opens Models folder |
| "Start writing workspace" | Opens Notion, Safari |
| "Start research workspace" | Opens Safari, Notion, ChatGPT |

### During Workspace

| Say this | What happens |
|----------|--------------|
| "Take notes" | Capture a voice note |
| "Brain dump" | Freeform idea capture (say "stop" when done) |
| "Add task" | Add to project task list |
| "Research [topic]" | Research and summarize a topic |
| "Summarize notes" | Summarize current notes |
| "Summarize session" | Create session summary |
| "End workspace" | Close session, auto-summarize |

## Configuration

Workspace profiles are stored in `~/.argus/workspaces.json`. You can edit this file to:

- Add custom profiles
- Change which apps launch
- Modify trigger phrases
- Add custom scripts

### Example: Custom Profile

```json
{
  "music_production": {
    "name": "Music Production",
    "description": "DAW and audio workspace",
    "trigger_phrases": ["music", "production", "audio", "daw", "recording"],
    "apps": [
      {"name": "Logic Pro", "type": "app"},
      {"name": "Spotify", "type": "app"}
    ],
    "directories": ["~/Music/Projects"],
    "sensory_mode": "disabled",
    "startup_delay": 2.0
  }
}
```

### App Types

```json
// Regular macOS/Windows app
{"name": "Visual Studio Code", "type": "app"}

// App opened to specific folder
{"name": "Finder", "type": "app", "path": "~/Projects"}

// Python script (like HELOSFORGE)
{
  "name": "HELOSFORGE",
  "type": "python_script",
  "path": "~/HELOSFORGE/HELOSFORGE.py",
  "args": ["--some-flag"]
}

// Shell script
{
  "name": "Setup Script",
  "type": "shell_script", 
  "path": "~/scripts/setup.sh"
}
```

## Sensory Modes

Each workspace can specify how Sensory_System should behave:

| Mode | Behavior |
|------|----------|
| `normal` | Full vision processing (face/object detection) |
| `low_power` | Reduced processing (presence detection only) |
| `disabled` | Vision completely off (frees GPU for other apps) |

**3D Modeling uses `disabled`** because HELOSFORGE needs the camera for gesture control.

## Project Structure

```
~/ARGUSWorkspace/
└── projects/
    └── my_project/
        ├── notes_2024-01-15_10-30.md
        ├── notes_2024-01-16_14-00.md
        └── tasks.md
```

### Notes Format

```markdown
# My Project

**Profile:** 3D Modeling
**Started:** 2024-01-15 10:30
**Session ID:** a1b2c3d4

## Project Summary

Building a custom phone stand with cable management.

---

- [10:35] Need to consider cable routing
- [10:42] Maximum height should be 15cm

## Brain Dump — 2024-01-15 10:45

- Could add a wireless charging coil
- Look into flexible mounting arms
- Maybe integrate with desk lamp

## Session Summary

Designed initial phone stand concept focusing on...

---
Session ended: 2024-01-15 12:00
```

## Files

| File | Purpose |
|------|---------|
| `workspace_config.py` | Profile management, persistence |
| `workspace_manager.py` | Core workspace logic |
| `workspace_integration.py` | ARGUS integration helpers |
| `main.py` | CLI entry point, backwards-compatible class |
| `workspaces_example.json` | Example configuration |

## Setup for HELOSFORGE

1. Make sure HELOSFORGE is installed:
   ```bash
   git clone https://github.com/BJW333/HELOSFORGE.git ~/HELOSFORGE
   ```

2. Edit `~/.argus/workspaces.json` and update the HELOSFORGE path:
   ```json
   "path": "~/HELOSFORGE/HELOSFORGE.py"
   ```

3. The 3D modeling workspace will:
   - Disable Sensory_System (frees the camera)
   - Launch HELOSFORGE
   - Open your Models folder
   - Start Spotify

## Future: Sensory_System Integration

When we optimize Sensory_System for low-power mode, workspaces will be able to:

1. **Coding workspace** → Keep lightweight presence detection, disable face recognition
2. **Research workspace** → Full awareness (knows who's at your desk)
3. **3D modeling workspace** → Completely off (HELOSFORGE needs camera)

This is controlled by the `sensory_mode` field in each profile.

## Troubleshooting

### Apps not opening

- Check the app name matches exactly (case-sensitive on some systems)
- For custom scripts, ensure the path is correct and script is executable

### HELOSFORGE not launching

- Verify the path in `~/.argus/workspaces.json`
- Make sure Python dependencies are installed
- Check that no other app is using the camera

### Summarizer errors

- Install transformers: `pip install transformers tensorflow`
- First run downloads ~500MB of model data

## Contributing

This is part of the ARGUS ecosystem. Feel free to extend with:

- New workspace profiles
- Additional voice commands  
- Integration with other tools

---

## Changelog (Fixed Version)

### Bug Fixes

1. **Brain dump infinite loop** - Added max empty counter to prevent infinite loops when `listen()` returns empty
2. **"stop" detection too broad** - Now checks for standalone stop phrases, not substrings (e.g., "I need to stop procrastinating" won't end the dump)
3. **Missing workspace checks** - Added checks in `take_notes()` and `brain_dump()` 
4. **Research not functional** - Added `research_fn` parameter for actual research capability
5. **Sensory control can't restart** - Added `sensory_start_fn` parameter to enable restart after stop
6. **"research" command conflict** - Fixed so "research workspace" doesn't trigger `research_topic()`
7. **transformers import crash** - Now uses lazy import so missing package doesn't crash
8. **Short trigger false matches** - "3d" now requires word boundary to avoid matching "started"
9. **JSON validation missing** - Added profile validation when loading from config
10. **File encoding issues** - Added `encoding="utf-8"` to all file operations

### New Features

- `list_projects()` - List all existing projects
- `switch_project()` - Switch to a different project
- `workspace_status()` - Report current workspace status
- `reset_to_defaults()` - Reset configuration
- Profile validation with `validate()` method
- Corrupted config file backup
