import platform
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import re
import threading 

from .workspace_config import (
    WorkspaceProfile,
    ActiveWorkspace,
    WorkspaceConfigManager,
    get_config_manager,
)
from config_metrics.main_config import WORKSPACE_ROOT, PROJECTS_ROOT

from actions.actions import find_app_path
from config_metrics.logging import log_debug

from speech.listen import pause_wake_listener, resume_wake_listener

class WorkspaceManager:
    """
    Manages workspace sessions with natural voice integration.
    Jarvis-style: brief, confident, helpful.
    """
    
    # Stop phrases for brain dump
    STOP_PHRASES = ["stop", "done", "that's all", "that's it", "finished", "end", "quit"]
    
    def __init__(
        self,
        speak_fn: Optional[Callable[[str], None]] = None,
        listen_fn: Optional[Callable[[], str]] = None,
        summarize_fn: Optional[Callable[[str], str]] = None,
        research_fn: Optional[Callable[[str], str]] = None,
        sensory_control_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the workspace manager.
        
        Args:
            speak_fn: Function to speak text (TTS)
            listen_fn: Function to listen for voice input
            summarize_fn: Function to summarize text
            research_fn: Function to research a topic
            sensory_control_fn: Function to control Sensory_System mode
        """
        self.speak = speak_fn or (lambda x: print(f"[SPEAK] {x}"))
        self.listen = listen_fn or (lambda: input("[LISTEN] > "))
        self.summarize = summarize_fn
        self.research = research_fn
        self.sensory_control = sensory_control_fn
        
        self.config = get_config_manager()
        self.current_workspace: Optional[ActiveWorkspace] = None
        
        # Threading lock for workspace operations
        self._ws_thread_lock = threading.Lock()  # Create the lock FIRST
        self._ws_thread_active = False
        self._ws_thread = None
        
        # Restore active workspace if exists
        if self.config.active_workspace:
            self.current_workspace = self.config.active_workspace
            log_debug(f"Restored active workspace: {self.current_workspace.project_name}")
        
        
    # =========================================================================
    # SPEECH HELPERS
    # =========================================================================
    
    def _wait_for_speech(self):
        """Wait for speech queue to finish."""
        try:
            from speech.speechmanager import speech_manager
            while not speech_manager.speech_queue.empty() or speech_manager.speaking:
                time.sleep(0.1)
            time.sleep(0.15)
        except ImportError:
            time.sleep(2.0)
    
    def prompt(self, question: str, fallback: str = "") -> str:
        """Ask a question and get voice response."""
        self.speak(question)
        self._wait_for_speech()
        
        try:
            answer = self.listen()
            return answer.strip() if answer else fallback
        except Exception as e:
            log_debug(f"Listen failed: {e}")
            return fallback
    
    def _is_cancel(self, text: str) -> bool:
        """Check if user wants to cancel."""
        if not text or not text.strip():
            return False  # Empty = voice failed, not cancel
        cancel_words = ["cancel", "cancelled", "canceled", "never mind", "nevermind", "forget it"]
        return text.lower().strip() in cancel_words
    
    # =========================================================================
    # PUBLIC API - Voice Command Handler
    # =========================================================================
    
    def handle_voice_command(self, command: str) -> bool:
        """
        Handle a voice command related to workspaces.
        Returns True if command was handled, False otherwise.
        """
        cmd_lower = command.lower()
        
        # Start workspace commands
        if re.search(r"\b(start|open|launch)\b", cmd_lower) and re.search(r"\bwork\s*space\b", cmd_lower):
            profile = self.config.match_profile(cmd_lower)

            # Check for quick mode - skip project name question
            quick_mode = "quick" in cmd_lower or "fast" in cmd_lower
            
            # Run synchronously - needs microphone access
            with self._ws_thread_lock:
                if self._ws_thread_active:
                    self.speak("Already on it.")
                    return True
                self._ws_thread_active = True
            
            try:
                if profile:
                    if quick_mode:
                        self.start_workspace_quick(profile.name)
                    else:
                        self.start_workspace(profile.name)
                else:
                    self.start_workspace_interactive()
            finally:
                with self._ws_thread_lock:
                    self._ws_thread_active = False
                    
            return True
        
        # End workspace
        if re.search(r"\b(end|stop|close|exit)\b", cmd_lower) and re.search(r"\bwork\s*space\b", cmd_lower):
            self.end_workspace()
            return True
        
        # Resume workspace - multiple trigger patterns
        if re.search(r"\bresume\b", cmd_lower) and re.search(r"\bwork\s*space\b", cmd_lower):
            self.resume_workspace()
            return True
        
        # "continue", "pick up where I left off", "last project"
        if re.search(r"\b(continue|pick up|left off|last project|where was i)\b", cmd_lower):
            self.resume_workspace()
            return True
        
        # List projects
        if re.search(r"\b(list|show)\b", cmd_lower) and re.search(r"\bproject\b", cmd_lower):
            self.list_projects()
            return True
        
        # Switch project  
        if re.search(r"\b(switch|change)\b", cmd_lower) and re.search(r"\bproject\b", cmd_lower):
            self.switch_project()
            return True
        
        # Workspace status
        if re.search(r"\bwork\s*space\b", cmd_lower) and re.search(r"\b(status|what|where)\b", cmd_lower):
            self.workspace_status()
            return True
        
        #Active workspace commands
        if self.current_workspace:
            if "take note" in cmd_lower or "take a note" in cmd_lower or "note this" in cmd_lower:
                self.take_notes()
                return True
            elif "brain dump" in cmd_lower:
                self.brain_dump()
                return True
            elif re.search(r"\bresearch\s+(a\s+topic|topic|this|something)\b", cmd_lower) or cmd_lower.strip().startswith("research "):
                # Extract topic from command if present (e.g., "research quantum physics")
                topic = None
                if cmd_lower.strip().startswith("research "):
                    topic = cmd_lower.strip()[len("research "):].strip()
                    # Don't treat "research workspace" or meta-phrases as topics
                    if topic in ("topic", "this", "something", "a topic") or "workspace" in topic:
                        topic = None
                self.research_topic(topic=topic if topic else None)
                return True
            elif "add task" in cmd_lower or "new task" in cmd_lower or "task" in cmd_lower:
                self.add_task()
                return True
            elif "summarize" in cmd_lower:
                if "session" in cmd_lower:
                    self.summarize_session()
                else:
                    self.summarize_notes()
                return True
        
        return False
    
    # =========================================================================
    # WORKSPACE LIFECYCLE
    # =========================================================================
    
    def start_workspace(self, profile_name: str, project_name: Optional[str] = None):
        """Start a workspace with a specific profile."""
        profile = self.config.get_profile(profile_name.lower().replace(" ", "_"))
        if not profile:
            profile = self.config.match_profile(profile_name)
        
        if not profile:
            available = ", ".join(self.config.list_profiles())
            self.speak(f"Don't have that profile. Options are: {available}")
            return
        
        # Pause main listener - we need the mic
        pause_wake_listener()
        
        try:
            self.speak(f"{profile.name}. What are you working on?")
            self._wait_for_speech()
            
            project_input = self.listen()
            if self._is_cancel(project_input):
                self.speak("Cancelled.")
                return
            
            project_name = project_input.strip() if project_input and project_input.strip() else "Quick Session"
            
            # Confirm long project names
            if len(project_name) > 40:
                short_preview = project_name[:35] + "..."
                confirm = self.prompt(f"Got {short_preview}. Shorter name?")
                if confirm and not self._is_cancel(confirm) and confirm.lower() not in ["no", "nope", "that's fine", "it's fine", "fine", "ok", "okay"]:
                    project_name = confirm.strip()
      
            self._setup_project(profile, project_name, f"Working on: {project_name}")
            
            if self.sensory_control:
                try:
                    self.sensory_control(profile.sensory_mode)
                except Exception as e:
                    log_debug(f"Sensory control failed: {e}")
            
            self.speak("Setting up.")
            threading.Thread(target=self._launch_apps, args=(profile,), daemon=True).start()
            self.speak("Ready.")
        
        finally:
            resume_wake_listener()
            
    def start_workspace_quick(self, profile_name: str):
        """Start workspace immediately without asking for project name."""
        profile = self.config.get_profile(profile_name.lower().replace(" ", "_"))
        if not profile:
            profile = self.config.match_profile(profile_name)
        
        if not profile:
            available = ", ".join(self.config.list_profiles())
            self.speak(f"Don't have that profile. Options are: {available}")
            return
        
        # Auto-generate project name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        project_name = f"{profile.name} {timestamp}"
        
        self._setup_project(profile, project_name, f"Quick {profile.name} session")
        
        if self.sensory_control:
            try:
                self.sensory_control(profile.sensory_mode)
            except Exception as e:
                log_debug(f"Sensory control failed: {e}")
        
        self.speak("Ready.")
        threading.Thread(target=self._launch_apps, args=(profile,), daemon=True).start()
        
    def start_workspace_interactive(self):
        """Start workspace with profile selection."""
        profiles = [p.name for p in self.config.profiles.values()]
        
        pause_wake_listener()
        
        try:
            choice = self.prompt(f"What kind of session? {', '.join(profiles)}")
            
            if self._is_cancel(choice):
                self.speak("Cancelled.")
                return
            
            profile = self.config.match_profile(choice)
            profile_to_use = profile if profile else self.config.get_profile("general")
            
            if not profile:
                self.speak("Didn't catch that. Going with general.")
            
            self.speak(f"{profile_to_use.name}. What are you working on?")
            self._wait_for_speech()
            
            project_input = self.listen()
            project_name = project_input.strip() if project_input and project_input.strip() else "Quick Session"
            
            #Confirm long project names
            if len(project_name) > 40:
                short_preview = project_name[:35] + "..."
                confirm = self.prompt(f"Got {short_preview}. Shorter name?")
                if confirm and not self._is_cancel(confirm) and confirm.lower() not in ["no", "nope", "that's fine", "it's fine", "fine", "ok", "okay"]:
                    project_name = confirm.strip()
            
            self._setup_project(profile_to_use, project_name, f"Working on: {project_name}")
            
            if self.sensory_control:
                try:
                    self.sensory_control(profile_to_use.sensory_mode)
                except Exception as e:
                    log_debug(f"Sensory control failed: {e}")
            
            self.speak("Setting up.")
            threading.Thread(target=self._launch_apps, args=(profile_to_use,), daemon=True).start()
            self.speak("Ready.")
        
        finally:
            resume_wake_listener()
    
    def end_workspace(self):
        """End the current workspace session."""
        if not self.current_workspace:
            self.speak("No active workspace.")
            return
        
        project_name = self.current_workspace.project_name
        
        # Quick end timestamp
        self._append_note(f"\n---\nSession ended: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        # Optional summary (silent - don't announce)
        if self.summarize:
            try:
                self._create_session_summary_silent()
            except Exception:
                pass
        
        # Re-enable Sensory_System
        if self.sensory_control:
            try:
                self.sensory_control("normal")
            except Exception as e:
                log_debug(f"Sensory restore failed: {e}")
        
        # Reset thread state
        with self._ws_thread_lock:
            self._ws_thread_active = False
        
        # Clear workspace
        self.current_workspace = None
        self.config.clear_active_workspace()
        
        # Simple confirmation
        self.speak(f"Closed {project_name}. Notes saved.")
    
    def resume_workspace(self):
        """Resume the last active workspace."""
        if self.current_workspace:
            self.speak(f"Already in {self.current_workspace.project_name}.")
            return
        
        saved = self.config.get_active_workspace()
        if saved:
            if not saved.project_dir.exists():
                self.speak("That project folder's gone.")
                self.config.clear_active_workspace()
                return
            
            self.current_workspace = saved
            
            # Apply sensory mode
            profile = self.config.get_profile(saved.profile_name)
            if profile and self.sensory_control:
                try:
                    self.sensory_control(profile.sensory_mode)
                except Exception:
                    pass
            
            self.speak(f"Resuming {saved.project_name}.")
        else:
            self.speak("Nothing to resume.")
    
    def workspace_status(self):
        """Report current workspace status - brief."""
        if self.current_workspace:
            ws = self.current_workspace
            duration = datetime.now() - ws.started_at
            minutes = int(duration.total_seconds() // 60)
            
            if minutes < 60:
                time_str = f"{minutes} minutes"
            else:
                hours = minutes // 60
                mins = minutes % 60
                time_str = f"{hours} hour{'s' if hours > 1 else ''}" + (f" {mins} min" if mins else "")
            
            self.speak(f"{ws.project_name}. {ws.profile_name} mode. {time_str} in.")
        else:
            self.speak("No active workspace.")
    
    def list_projects(self):
        """List existing projects - brief."""
        if not PROJECTS_ROOT.exists():
            self.speak("No projects yet.")
            return
        
        projects = sorted([d.name for d in PROJECTS_ROOT.iterdir() if d.is_dir()])
        if projects:
            count = len(projects)
            shown = projects[:5]
            names = ", ".join(p.replace("_", " ") for p in shown)
            
            if count <= 5:
                self.speak(f"{count} projects: {names}.")
            else:
                self.speak(f"{count} projects. Recent: {names}.")
        else:
            self.speak("No projects yet.")
    
    def switch_project(self):
        """Switch to a different project."""
        if not PROJECTS_ROOT.exists():
            self.speak("No projects.")
            return
        
        projects = [d.name for d in PROJECTS_ROOT.iterdir() if d.is_dir()]
        if not projects:
            self.speak("No projects.")
            return
        
        shown = projects[:5]
        project_name = self.prompt(f"Which one? {', '.join(p.replace('_', ' ') for p in shown)}")
        
        if self._is_cancel(project_name):
            self.speak("Cancelled.")
            return
        
        # Find matching project
        project_slug = self._slugify(project_name)
        project_dir = PROJECTS_ROOT / project_slug
        
        if not project_dir.exists():
            # Fuzzy match
            for p in projects:
                if project_name.lower() in p.lower():
                    project_dir = PROJECTS_ROOT / p
                    project_slug = p
                    break
        
        if not project_dir.exists():
            self.speak(f"Can't find {project_name}.")
            return
        
        # End current if active
        if self.current_workspace:
            self._append_note(f"\n---\nSwitched away: {datetime.now().strftime('%H:%M')}\n")
            self.current_workspace = None
            self.config.clear_active_workspace()
        
        # Get or create notes
        notes_files = sorted(project_dir.glob("notes_*.md"), reverse=True)
        notes_path = notes_files[0] if notes_files else project_dir / f"notes_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
        
        # Quick profile selection
        profiles = [p.name for p in self.config.profiles.values()]
        profile_choice = self.prompt(f"Session type? {', '.join(profiles)}")
        profile = self.config.match_profile(profile_choice) or self.config.get_profile("general")
        
        # Create workspace
        self.current_workspace = ActiveWorkspace(
            profile_name=profile.name.lower().replace(" ", "_"),
            project_name=project_name,
            project_slug=project_slug,
            project_dir=project_dir,
            notes_path=notes_path,
            started_at=datetime.now(),
            session_id=str(uuid.uuid4())[:8],
        )
        
        self.config.set_active_workspace(self.current_workspace)
        
        # Sensory mode
        if self.sensory_control:
            try:
                self.sensory_control(profile.sensory_mode)
            except Exception:
                pass
        
        # Launch apps
        self._launch_apps(profile)
        
        self.speak(f"Switched to {project_name}.")
    
    # =========================================================================
    # WORKSPACE COMMANDS - Natural Flow
    # =========================================================================
    
    def take_notes(self):
        """Quick note capture."""
        if not self.current_workspace:
            self.speak("Start a workspace first.")
            return
        
        note = self.prompt("Note?")
        
        if self._is_cancel(note):
            self.speak("Cancelled.")
            return
        
        timestamp = datetime.now().strftime("%H:%M")
        self._append_note(f"- [{timestamp}] {note}")
        self.speak("Done.")
    
    def brain_dump(self):
        """Freeform brain dump - continuous listening."""
        if not self.current_workspace:
            self.speak("Start a workspace first.")
            return
        
        self.speak("Go ahead.")
        self._wait_for_speech()
        
        chunks = []
        empty_count = 0
        max_empty = 3
        
        while True:
            chunk = self.listen()
            
            if not chunk or not chunk.strip():
                empty_count += 1
                if empty_count >= max_empty:
                    break
                # Silent retry - no nagging
                continue
            
            empty_count = 0
            chunk_lower = chunk.lower().strip()
            
            # Check for stop
            is_stop = any(
                chunk_lower == phrase or chunk_lower.startswith(phrase + " ")
                for phrase in self.STOP_PHRASES
            )
            
            if is_stop:
                break
            
            chunks.append(chunk.strip())
        
        if chunks:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            section = f"\n## Brain Dump - {timestamp}\n\n"
            section += "\n".join(f"- {line}" for line in chunks)
            section += "\n"
            self._append_note(section)
            self.speak(f"Got {len(chunks)} items.")
        else:
            self.speak("Nothing captured.")
    
    def research_topic(self, topic: Optional[str] = None):
        """Research a topic."""
        if not self.current_workspace:
            self.speak("Start a workspace first.")
            return
        
        if not topic:
            topic = self.prompt("What topic?")
        
        if self._is_cancel(topic):
            self.speak("Cancelled.")
            return
        
        self.speak("On it.")
        
        raw_info = None
        if self.research:
            try:
                raw_info = self.research(topic)
            except Exception as e:
                log_debug(f"Research failed: {e}")
        
        if raw_info:
            # Summarize if possible
            if self.summarize:
                try:
                    summary = self.summarize(raw_info)
                except Exception:
                    summary = raw_info[:500] + "..." if len(raw_info) > 500 else raw_info
            else:
                summary = raw_info[:500] + "..." if len(raw_info) > 500 else raw_info
            
            self._append_note(f"\n## Research: {topic}\n\n{summary}\n")
            self.speak("Added to notes.")
        else:
            self._append_note(f"\n## Research: {topic}\n\n[TODO]\n")
            self.speak("Added placeholder.")
    
    def add_task(self):
        """Add a task."""
        if not self.current_workspace:
            self.speak("Start a workspace first.")
            return
        
        task = self.prompt("What task?")
        
        if self._is_cancel(task):
            self.speak("Cancelled.")
            return
        
        tasks_path = self.current_workspace.project_dir / "tasks.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if not tasks_path.exists():
            with tasks_path.open("w", encoding="utf-8") as f:
                f.write(f"# Tasks - {self.current_workspace.project_name}\n\n")
        
        with tasks_path.open("a", encoding="utf-8") as f:
            f.write(f"- [ ] [{timestamp}] {task}\n")
        
        self.speak("Added.")
    
    def summarize_notes(self):
        """Summarize current notes."""
        if not self.current_workspace:
            self.speak("No active workspace.")
            return
        
        if not self.summarize:
            self.speak("Summarizer not available.")
            return
        
        notes_path = self.current_workspace.notes_path
        if not notes_path.exists():
            self.speak("No notes yet.")
            return
        
        text = notes_path.read_text(encoding="utf-8")
        if not text.strip():
            self.speak("Notes are empty.")
            return
        
        self.speak("Summarizing.")
        try:
            summary = self.summarize(text)
            self._append_note(f"\n## Summary\n\n{summary}\n")
            self.speak("Done.")
        except Exception as e:
            log_debug(f"Summarization failed: {e}")
            self.speak("Couldn't summarize.")
    
    def summarize_session(self):
        """Summarize the session."""
        if not self.current_workspace:
            self.speak("No active workspace.")
            return
        
        notes_path = self.current_workspace.notes_path
        if not notes_path.exists():
            self.speak("No notes yet.")
            return
        
        text = notes_path.read_text(encoding="utf-8")
        if not text.strip():
            self.speak("Notes are empty.")
            return
        
        if self.summarize:
            self.speak("Creating summary.")
            try:
                summary = self.summarize(text)
                self._append_note(f"\n## Session Summary\n\n{summary}\n")
                self.speak("Done.")
            except Exception as e:
                log_debug(f"Summarization failed: {e}")
                self.speak("Couldn't summarize.")
        else:
            self.speak("Summarizer not available.")
    
    def _create_session_summary_silent(self):
        """Create summary without speaking."""
        if not self.current_workspace or not self.summarize:
            return
        
        notes_path = self.current_workspace.notes_path
        if not notes_path.exists():
            return
        
        text = notes_path.read_text(encoding="utf-8")
        if not text.strip():
            return
        
        try:
            summary = self.summarize(text)
            self._append_note(f"\n## Session Summary\n\n{summary}\n")
        except Exception:
            pass
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _setup_project(self, profile: WorkspaceProfile, name: str, info: str):
        """Set up project directory and notes file."""
        project_name = name.strip() or "Quick Session"
        project_slug = self._slugify(project_name)
        
        project_dir = PROJECTS_ROOT / project_slug
        project_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        notes_path = project_dir / f"notes_{timestamp}.md"
        
        self.current_workspace = ActiveWorkspace(
            profile_name=profile.name.lower().replace(" ", "_"),
            project_name=project_name,
            project_slug=project_slug,
            project_dir=project_dir,
            notes_path=notes_path,
            started_at=datetime.now(),
            session_id=str(uuid.uuid4())[:8],
        )
        
        self.config.set_active_workspace(self.current_workspace)
        
        # Write header
        header = (
            f"# {project_name}\n\n"
            f"**Profile:** {profile.name}\n"
            f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"---\n"
        )
        self._append_note(header)
    
    def _append_note(self, text: str):
        """Append text to notes file."""
        if not self.current_workspace:
            return
        
        notes_path = self.current_workspace.notes_path
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with notes_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            log_debug(f"Failed to write notes: {e}")
    
    def _launch_apps(self, profile: WorkspaceProfile):
        """Launch profile apps - silent."""
        for app_config in profile.apps:
            try:
                app_type = app_config.get("type", "app")
                app_name = app_config.get("name", "Unknown")
                
                if app_type == "app":
                    self._open_app(app_name, app_config.get("path"))
                elif app_type == "python_script":
                    self._run_python_script(app_config)
                elif app_type == "shell_script":
                    self._run_shell_script(app_config)
                
                time.sleep(profile.startup_delay)
                
            except Exception as e:
                log_debug(f"Failed to launch {app_config}: {e}")
        
        # Open directories
        for dir_path in profile.directories:
            try:
                self._open_directory(Path(dir_path).expanduser())
            except Exception as e:
                log_debug(f"Failed to open directory {dir_path}: {e}")
    
    def _open_app(self, app_name: str, path: Optional[str] = None):
        """Open an application silently."""
        system = platform.system()
        
        try:
            if system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            elif system == "Windows":
                subprocess.Popen(["start", "", app_name], shell=True)
            elif system == "Linux":
                subprocess.Popen([app_name])
        except Exception:
            found = find_app_path(app_name)
            if found:
                subprocess.Popen(["open", found] if system == "Darwin" else [found])
        
        # If path provided, open that directory too
        if path:
            self._open_directory(Path(path).expanduser())
    
    def _run_python_script(self, config: Dict[str, Any]):
        """Run Python script."""
        script_path = Path(config["path"]).expanduser()
        args = config.get("args", [])
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        subprocess.Popen(
            ["python3", str(script_path)] + args,
            cwd=script_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log_debug(f"Launched: {script_path}")
    
    def _run_shell_script(self, config: Dict[str, Any]):
        """Run shell script."""
        script_path = Path(config["path"]).expanduser()
        args = config.get("args", [])
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        subprocess.Popen(["bash", str(script_path)] + args, cwd=script_path.parent)
        log_debug(f"Launched: {script_path}")
    
    def _open_directory(self, path: Path):
        """Open directory in file browser."""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", str(path)])
            elif system == "Windows":
                subprocess.Popen(["explorer", str(path)])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            log_debug(f"Failed to open directory {path}: {e}")
    
    @staticmethod
    def _slugify(name: str) -> str:
        """Convert name to filesystem-safe slug."""
        return (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "")
            .replace("'", "")
            .replace('"', "")
        )


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def create_workspace_manager(
    speak_fn=None,
    listen_fn=None,
    summarize_fn=None,
    research_fn=None,
    sensory_hub=None,
    sensory_start_fn=None,
) -> WorkspaceManager:
    """Create a WorkspaceManager configured for ARGUS."""
    sensory_control = None
    
    if sensory_hub is not None:
        _hub_state = {"hub": sensory_hub, "stopped": False}
        
        def control_sensory(mode: str):
            if mode == "disabled":
                if _hub_state["hub"] and not _hub_state["stopped"]:
                    _hub_state["hub"].stop()
                    _hub_state["stopped"] = True
            elif mode == "low_power":
                log_debug("Low-power mode not implemented")
            elif mode == "normal":
                if _hub_state["stopped"] and sensory_start_fn:
                    try:
                        _hub_state["hub"] = sensory_start_fn()
                        _hub_state["stopped"] = False
                    except Exception as e:
                        log_debug(f"Sensory restart failed: {e}")
        
        sensory_control = control_sensory
    
    return WorkspaceManager(
        speak_fn=speak_fn,
        listen_fn=listen_fn,
        summarize_fn=summarize_fn,
        research_fn=research_fn,
        sensory_control_fn=sensory_control,
    )


# =========================================================================
# CLI TESTING
# =========================================================================

if __name__ == "__main__":    
    manager = WorkspaceManager()
    
    print("\nAvailable profiles:")
    for name in manager.config.list_profiles():
        profile = manager.config.get_profile(name)
        print(f"  - {profile.name}: {profile.trigger_phrases}")
    
    print("\nTest commands:")
    test_commands = [
        "start coding workspace",
        "start 3d modeling workspace",
        "end workspace",
    ]
    
    for cmd in test_commands:
        profile = manager.config.match_profile(cmd)
        print(f"  '{cmd}' -> profile={profile.name if profile else 'None'}")
