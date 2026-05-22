"""
ARGUS Workspace Integration
===========================

This module shows how to integrate the Workspace system with ARGUS.
Drop this into your ARGUS project and call `handle_workspace_command()`
when workspace-related voice commands are detected.

CHANGELOG:
- Fixed: transformers import is now lazy (won't crash if not installed)
- Added: research_fn parameter support
- Added: sensory_start_fn for restart capability

Example integration in ARGUS main.py:

    from workspace_integration import ArgusWorkspaceIntegration
    
    # Initialize once at startup
    workspace = ArgusWorkspaceIntegration(
        speak_fn=speak,
        listen_fn=generalvoiceinput,
    )
    
    # In your command handler:
    def handle_command(command: str):
        # Check if it's a workspace command first
        if workspace.handle_command(command):
            return  # Workspace handled it
        
        # ... rest of your command handling
"""

from typing import Callable, Optional

from .workspace_manager import create_workspace_manager, WorkspaceManager
from .workspace_config import get_config_manager, WorkspaceConfigManager
from config_metrics.logging import log_debug    


class ArgusWorkspaceIntegration:
    """
    High-level integration class for ARGUS.
    
    Handles workspace voice commands and manages the workspace lifecycle.
    """
    
    # Voice triggers that indicate a workspace command
    WORKSPACE_TRIGGERS = [
        "start workspace",
        "start my workspace",
        "begin workspace",
        "open workspace",
        "coding workspace",
        "open my workspace",
        "close my workspace",
        "3d modeling workspace",
        "modeling workspace",
        "writing workspace",
        "research workspace",
        "brainstorm workspace",
        "end workspace",
        "close workspace",
        "stop workspace",
        "resume workspace",
        "workspace status",
        "list projects",
        "switch project",
        # Commands when workspace is active
        "take notes",
        "take a note",
        "brain dump",
        "add task",
        "new task",
        "summarize notes",
        "summarize session",
    ]
    
    def __init__(
        self,
        speak_fn: Callable[[str], None],
        listen_fn: Callable[[], str],
        research_fn: Optional[Callable[[str], str]] = None,
        sensory_hub=None,
        sensory_start_fn=None,
        use_summarizer: bool = True,
    ):
        """
        Initialize ARGUS workspace integration.
        
        Args:
            speak_fn: Your ARGUS speak() function
            listen_fn: Your ARGUS generalvoiceinput() function
            research_fn: Function to research topics (e.g., gatherinfofromknowledgebase)
            sensory_hub: Optional Sensory_System hub for power management
            sensory_start_fn: Function to restart sensory hub if stopped
            use_summarizer: Whether to load the T5 summarizer (slower startup)
        """
        self.speak = speak_fn
        self.listen = listen_fn
        
        # Lazy-load summarizer to avoid slow startup
        self._summarizer = None
        self._use_summarizer = use_summarizer
        self._summarizer_failed = False
        
        # Create the workspace manager
        self.manager = create_workspace_manager(
            speak_fn=speak_fn,
            listen_fn=listen_fn,
            summarize_fn=self._summarize if use_summarizer else None,
            research_fn=research_fn,
            sensory_hub=sensory_hub,
            sensory_start_fn=sensory_start_fn,
        )
        
        log_debug("ARGUS Workspace Integration initialized")
    
    def is_workspace_command(self, command: str) -> bool:
        """
        Check if a command is workspace-related.
        Use this to decide whether to route to workspace handler.
        """
        cmd_lower = command.lower()
        
        # Direct triggers
        for trigger in self.WORKSPACE_TRIGGERS:
            if trigger in cmd_lower:
                return True
        
        # If workspace is active, more commands are valid
        if self.manager.current_workspace:
            workspace_active_triggers = [
                "take note", "take a note", "note this",
                "add task", "new task",
                "brain dump",
                "summarize",
            ]
            for trigger in workspace_active_triggers:
                if trigger in cmd_lower:
                    return True
            
            # "research <topic>" but NOT "research workspace" or bare "research"
            if cmd_lower.strip().startswith("research ") and "workspace" not in cmd_lower:
                return True
        
        return False
    
    def handle_command(self, command: str) -> bool:
        """
        Handle a workspace command.
        
        Returns:
            True if command was handled, False if not a workspace command
        """
        if not self.is_workspace_command(command):
            return False
        
        return self.manager.handle_voice_command(command)
    
    def get_status(self) -> dict:
        """Get current workspace status for ARGUS context."""
        if self.manager.current_workspace:
            ws = self.manager.current_workspace
            return {
                "active": True,
                "project_name": ws.project_name,
                "profile": ws.profile_name,
                "session_id": ws.session_id,
                "started_at": ws.started_at.isoformat(),
            }
        return {"active": False}
    
    def _get_summarizer(self):
        """Lazy-load the summarizer."""
        if self._summarizer_failed:
            return None
        
        if self._summarizer is None and self._use_summarizer:
            try:
                #Lazy import to avoid crash if transformers not installed
                from transformers import pipeline
                
                #Try PyTorch first (more common in Ollama setups) fall back to TF
                for fw in ("pt", "tf"):
                    try:
                        self._summarizer = pipeline(
                            "summarization",
                            model="google-t5/t5-base",
                            tokenizer="google-t5/t5-base",
                            framework=fw,
                        )
                        log_debug(f"Summarizer loaded (framework={fw})")
                        break
                    except (ImportError, RuntimeError):
                        continue
                
                if self._summarizer is None:
                    raise ImportError("Neither PyTorch nor TensorFlow available")
            except ImportError:
                log_debug("transformers not installed - summarizer unavailable")
                self._summarizer_failed = True
                self._use_summarizer = False
            except Exception as e:
                log_debug(f"Failed to load summarizer: {e}")
                self._summarizer_failed = True
                self._use_summarizer = False
        
        return self._summarizer
    
    def _summarize(self, text: str) -> str:
        """Summarize text using T5."""
        summarizer = self._get_summarizer()
        if not summarizer:
            return "[Summarizer unavailable - install transformers]"
        
        # Truncate if too long
        max_chars = 6000
        if len(text) > max_chars:
            text = text[-max_chars:]
        
        try:
            result = summarizer(text, min_length=5, max_length=500)
            return result[0].get("summary_text", "").strip()
        except Exception as e:
            log_debug(f"Summarization failed: {e}")
            return "[Summarization failed]"


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def quick_start_workspace(speak_fn, listen_fn, research_fn=None):
    """
    Quickly start a workspace with minimal setup.
    
    Usage:
        from workspace_integration import quick_start_workspace
        quick_start_workspace(speak, generalvoiceinput)
    """
    integration = ArgusWorkspaceIntegration(
        speak_fn=speak_fn,
        listen_fn=listen_fn,
        research_fn=research_fn,
        use_summarizer=False,  # Faster startup
    )
    integration.manager.start_workspace_interactive()
    return integration


# =============================================================================
# EXAMPLE ARGUS INTEGRATION
# =============================================================================

def example_argus_integration():
    """
    Example showing how to integrate with ARGUS.
    
    This is pseudocode showing the integration pattern.
    """
    
    # --- In your ARGUS main.py ---
    
    # 1. Import the integration
    # from workspace_integration import ArgusWorkspaceIntegration
    
    # 2. Initialize at startup (after your speak/listen functions are ready)
    # workspace = ArgusWorkspaceIntegration(
    #     speak_fn=speak,
    #     listen_fn=generalvoiceinput,
    #     research_fn=gatherinfofromknowledgebase,  # Your research function
    #     sensory_hub=sensory_hub,  # Optional - for power management
    #     sensory_start_fn=lambda: start_sensor_hub(),  # Optional - for restart
    # )
    
    # 3. In your main command handler (after wake word detection):
    # def process_command(command: str):
    #     # Check workspace commands first
    #     if workspace.handle_command(command):
    #         return
    #     
    #     # ... rest of your command handling (web search, apps, etc.)
    
    # 4. Optionally, provide workspace context to your AI responses:
    # context = {
    #     "workspace": workspace.get_status(),
    #     "user_present": CONTEXT.get("user_present"),
    #     # ... other context
    # }
    
    print("See the docstring for integration example")


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    
    
    print("=" * 60)
    print("ARGUS Workspace Integration Demo")
    print("=" * 60)
    
    # Simple console-based demo
    def demo_speak(text):
        print(f"\n[ARGUS]: {text}")
    
    def demo_listen():
        return input("[You]: ").strip()
    
    def demo_research(topic):
        return f"Mock research results for: {topic}"
    
    integration = ArgusWorkspaceIntegration(
        speak_fn=demo_speak,
        listen_fn=demo_listen,
        research_fn=demo_research,
        use_summarizer=False,  # Skip for demo
    )
    
    print("\nAvailable workspace profiles:")
    for name in integration.manager.config.list_profiles():
        profile = integration.manager.config.get_profile(name)
        print(f"  * {profile.name} - triggers: {profile.trigger_phrases[:3]}...")
    
    print("\n" + "-" * 60)
    print("Try commands like:")
    print("  'start coding workspace'")
    print("  'start 3d modeling workspace'")
    print("  'take notes' (when workspace active)")
    print("  'brain dump' (when workspace active)")
    print("  'list projects'")
    print("  'workspace status'")
    print("  'end workspace'")
    print("  'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n[Command]: ").strip()
            
            if cmd.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if integration.handle_command(cmd):
                print("  [OK] Command handled by workspace")
            else:
                print("  [--] Not a workspace command")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
