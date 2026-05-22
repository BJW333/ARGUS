"""
ARGUS Workspace - Main Entry Point
==================================

This is an updated version of your original workspace main.py
that uses the new modular workspace system.

CHANGELOG:
- Fixed emoji encoding issues (using ASCII fallbacks)
- Added research_fn integration
- Added sensory restart capability
- Better error handling

Features:
- JSON-based configurable profiles
- Proper 3D modeling support (launches HELOSFORGE)
- Sensory_System power management hooks
- Persistent workspace state
- Cleaner architecture

Usage:
    python main.py                    # Interactive startup
    python main.py --profile coding   # Direct profile start
    python main.py --resume           # Resume last workspace
"""

import argparse
import sys
from pathlib import Path

# Add workspace modules to path if needed
WORKSPACE_MODULE_DIR = Path(__file__).parent
if str(WORKSPACE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_MODULE_DIR))

from workspace_manager import WorkspaceManager
from workspace_config import get_config_manager, WorkspaceProfile

from config_metrics.logging import log_debug

# =============================================================================
# CONFIGURE THESE IMPORTS FOR YOUR ARGUS SETUP
# =============================================================================

# Option 1: Import from your existing ARGUS
# from actions.actions import open_app, takenotes, gatherinfofromknowledgebase
# from speech.listen import generalvoiceinput
# from speech.speak import speak

# Option 2: Use fallbacks for standalone testing
try:
    from speech.speak import speak
except ImportError:
    def speak(text: str):
        # FIX: Use ASCII to avoid encoding issues
        print(f"[SPEAK] {text}")

try:
    from speech.listen import generalvoiceinput
except ImportError:
    def generalvoiceinput() -> str:
        return input("[LISTEN] > ").strip()

# Optional: Import research function
try:
    from actions.actions import gatherinfofromknowledgebase
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False
    gatherinfofromknowledgebase = None

# Optional: Import Sensory_System for power management
try:
    from sensory_system.main import start_sensor_hub
    SENSORY_AVAILABLE = True
except ImportError:
    SENSORY_AVAILABLE = False
    start_sensor_hub = None

# Optional: Import summarizer
try:
    from transformers import pipeline
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    pipeline = None


# =============================================================================
# WORKSPACE CLASS (Backwards Compatible)
# =============================================================================

class Workspace:
    """
    Backwards-compatible Workspace class that wraps the new WorkspaceManager.
    
    You can use this as a drop-in replacement for your original Workspace class,
    or switch to using WorkspaceManager directly for more control.
    """
    
    def __init__(self):
        self._summarizer = None
        self._sensory_hub = None
        
        # Create the workspace manager
        self.manager = WorkspaceManager(
            speak_fn=speak,
            listen_fn=generalvoiceinput,
            summarize_fn=self._summarize_text,
            research_fn=self._research_topic,
            sensory_control_fn=self._control_sensory,
        )
    
    def run(self, profile: str = None, topic: str = None):
        """
        Main entry point for starting a workspace session.
        
        Args:
            profile: Optional profile name to start directly
            topic: Optional topic to research at startup
        """
        if profile:
            # Direct profile start
            self.manager.start_workspace(profile)
        else:
            # Interactive profile selection
            self.manager.start_workspace_interactive()
        
        # Research initial topic if provided
        if topic and self.manager.current_workspace:
            self.manager.research_topic(topic)
        
        # Enter command loop
        self._command_loop()
    
    def resume(self):
        """Resume the last workspace session."""
        self.manager.resume_workspace()
        if self.manager.current_workspace:
            self._command_loop()
    
    def _command_loop(self):
        """Main interactive command loop."""
        if not self.manager.current_workspace:
            speak("No active workspace.")
            return
        
        speak(
            "I'm in workspace mode. "
            "Say things like 'take notes', 'brain dump', 'research', "
            "'add task', 'summarize session', or 'end workspace'."
        )
        
        while self.manager.current_workspace:
            try:
                choice = self.manager.prompt("What would you like to do?")
                
                # Handle the command
                handled = self.manager.handle_voice_command(choice)
                
                if not handled:
                    speak("I didn't catch that. Try again?")
                    
            except KeyboardInterrupt:
                speak("Interrupted. Ending workspace.")
                self.manager.end_workspace()
                break
    
    def _summarize_text(self, text: str) -> str:
        """Summarize text using T5 model."""
        if not SUMMARIZER_AVAILABLE:
            return "[Summarizer not available - install transformers]"
        
        if self._summarizer is None:
            try:
                self._summarizer = pipeline(
                    "summarization",
                    model="google-t5/t5-base",
                    tokenizer="google-t5/t5-base",
                    framework="tf",
                )
            except Exception as e:
                log_debug(f"Failed to load summarizer for workspace: {e}")
                return f"[Summarizer error: {e}]"
        
        # Truncate long text
        max_chars = 6000
        if len(text) > max_chars:
            text = text[-max_chars:]
        
        try:
            result = self._summarizer(text, min_length=5, max_length=500)
            return result[0].get("summary_text", "").strip()
        except Exception as e:
            return f"[Summarization failed: {e}]"
    
    def _research_topic(self, topic: str) -> str:
        """Research a topic using the knowledge base."""
        if not RESEARCH_AVAILABLE or not gatherinfofromknowledgebase:
            return None
        
        try:
            return gatherinfofromknowledgebase(topic)
        except Exception as e:
            log_debug(f"Research failed: {e}")
            return None
    
    def _control_sensory(self, mode: str):
        """Control Sensory_System power mode."""
        if not SENSORY_AVAILABLE:
            log_debug("Sensory_System not available")
            return
        
        if mode == "disabled":
            if self._sensory_hub:
                try:
                    self._sensory_hub.stop()
                    self._sensory_hub = None
                    log_debug("Sensory_System stopped")
                except Exception as e:
                    log_debug(f"Failed to stop Sensory_System: {e}")
        elif mode == "normal":
            if not self._sensory_hub:
                try:
                    self._sensory_hub = start_sensor_hub(
                        debug=False,
                        console_interval=None,
                    )
                    log_debug("Sensory_System started in normal mode")
                except Exception as e:
                    log_debug(f"Failed to start Sensory_System: {e}")
        elif mode == "low_power":
            # TODO: Implement low-power mode in Sensory_System
            log_debug("Low-power mode not yet implemented")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for ARGUS Workspace."""
    parser = argparse.ArgumentParser(
        description="ARGUS Workspace - Voice-driven project workspace manager"
    )
    parser.add_argument(
        "--profile", "-p",
        help="Start with specific profile (coding, 3d_modeling, writing, research, brainstorm)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume last workspace session",
    )
    parser.add_argument(
        "--topic", "-t",
        help="Research topic to add at startup",
    )
    parser.add_argument(
        "--list-profiles", "-l",
        action="store_true",
        help="List available workspace profiles",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    
    # List profiles and exit
    if args.list_profiles:
        config = get_config_manager()
        print("\nAvailable Workspace Profiles:")
        print("-" * 40)
        for name in config.list_profiles():
            profile = config.get_profile(name)
            print(f"\n  {profile.name}")
            print(f"    Triggers: {', '.join(profile.trigger_phrases[:4])}")
            print(f"    Apps: {', '.join(a['name'] for a in profile.apps)}")
            print(f"    Sensory mode: {profile.sensory_mode}")
        return
    
    # Create and run workspace
    workspace = Workspace()
    
    if args.resume:
        workspace.resume()
    else:
        workspace.run(profile=args.profile, topic=args.topic)


# =============================================================================
# ARGUS INTEGRATION EXAMPLE
# =============================================================================

def integrate_with_argus():
    """
    Example showing how to integrate with your ARGUS main command loop.
    
    In your ARGUS main.py, after wake word detection:
    
        from services.workspace.main import workspace_handler
        
        def handle_command(command):
            # Check workspace commands first
            if workspace_handler.handle_command(command):
                return
            
            # ... rest of your command handling
    """
    pass


# Global workspace handler for ARGUS integration
workspace_handler = None

def get_workspace_handler() -> Workspace:
    """Get or create the global workspace handler."""
    global workspace_handler
    if workspace_handler is None:
        workspace_handler = Workspace()
    return workspace_handler


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
