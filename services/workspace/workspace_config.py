import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from config_metrics.logging import log_debug

from config_metrics.main_config import (
    WORKSPACE_CONFIG_DIR as CONFIG_DIR,
    WORKSPACES_CONFIG_FILE as WORKSPACES_CONFIG,
    WORKSPACE_ROOT,
    PROJECTS_ROOT,
)


@dataclass
class WorkspaceProfile:
    """Defines a workspace profile configuration."""
    name: str
    description: str
    trigger_phrases: List[str]  # Voice triggers like "3d modeling", "coding"
    apps: List[Dict[str, Any]]  # Apps to launch
    directories: List[str] = field(default_factory=list)  # Dirs to open
    scripts: List[Dict[str, Any]] = field(default_factory=list)  # Custom scripts
    environment: Dict[str, str] = field(default_factory=dict)  # Env vars
    sensory_mode: str = "normal"  # "normal", "low_power", "disabled"
    startup_delay: float = 1.0  # Delay between app launches
    
    def matches_input(self, user_input: str) -> bool:
        """
        Check if user input matches this profile's triggers.
        
        FIX: For short triggers (<=3 chars), require word boundary to avoid
        false matches like "3d" matching "started".
        """
        user_lower = user_input.lower()
        
        for phrase in self.trigger_phrases:
            phrase_lower = phrase.lower()
            
            if len(phrase_lower) <= 3:
                # Short phrase - require word boundary
                pattern = rf'\b{re.escape(phrase_lower)}\b'
                if re.search(pattern, user_lower):
                    return True
            else:
                # Longer phrase - simple substring match is fine
                if phrase_lower in user_lower:
                    return True
        
        return False
    
    def validate(self) -> List[str]:
        """
        Validate the profile configuration.
        Returns list of error messages (empty if valid).
        """
        errors = []
        
        if not self.name:
            errors.append("Profile name is required")
        
        if not self.trigger_phrases:
            errors.append("At least one trigger phrase is required")
        
        valid_modes = ["normal", "low_power", "disabled"]
        if self.sensory_mode not in valid_modes:
            errors.append(f"Invalid sensory_mode '{self.sensory_mode}', must be one of {valid_modes}")
        
        if self.startup_delay < 0:
            errors.append("startup_delay must be non-negative")
        
        for i, app in enumerate(self.apps):
            if not isinstance(app, dict):
                errors.append(f"App #{i} is not a dict: {app}")
                continue
            if "name" not in app:
                errors.append(f"App #{i} missing 'name' field: {app}")
            if app.get("type") == "python_script" and "path" not in app:
                errors.append(f"Python script app #{i} missing 'path' field: {app}")
        
        return errors


@dataclass
class ActiveWorkspace:
    """Represents a currently active workspace session."""
    profile_name: str
    project_name: str
    project_slug: str
    project_dir: Path
    notes_path: Path
    started_at: datetime
    session_id: str
    
    def to_dict(self) -> dict:
        return {
            "profile_name": self.profile_name,
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "project_dir": str(self.project_dir),
            "notes_path": str(self.notes_path),
            "started_at": self.started_at.isoformat(),
            "session_id": self.session_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActiveWorkspace":
        return cls(
            profile_name=data["profile_name"],
            project_name=data["project_name"],
            project_slug=data["project_slug"],
            project_dir=Path(data["project_dir"]),
            notes_path=Path(data["notes_path"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            session_id=data["session_id"],
        )


def get_default_profiles() -> Dict[str, WorkspaceProfile]:
    """Return default workspace profiles."""
    return {
        "coding": WorkspaceProfile(
            name="Coding",
            description="Software development workspace",
            trigger_phrases=["coding", "programming", "development", "code session", "dev session"],
            apps=[
                {"name": "Visual Studio Code", "type": "app"},
                {"name": "Terminal", "type": "app"},
                {"name": "Notion", "type": "app"},
                {"name": "Spotify", "type": "app"},
            ],
            directories=["~/Projects"],
            sensory_mode="low_power",
        ),
        
        "3d_modeling": WorkspaceProfile(
            name="3D Modeling",
            description="3D modeling and CAD workspace with HELOSFORGE",
            trigger_phrases=["3d modeling", "3d model", "modeling", "cad", "helosforge", "forge"],
            apps=[
                {
                    "name": "HELOSFORGE",
                    "type": "python_script",
                    "path": "~/HELOSFORGE/HELOSFORGE.py",
                    "args": [],
                },
                {"name": "Finder", "type": "app", "path": "~/Models"},
                {"name": "Spotify", "type": "app"},
            ],
            directories=["~/Models", "~/HELOSFORGE/outputs"],
            sensory_mode="disabled",
            startup_delay=2.0,
        ),
        
        "writing": WorkspaceProfile(
            name="Writing",
            description="Writing and documentation workspace",
            trigger_phrases=["writing", "write session", "documentation", "docs", "blog", "article"],
            apps=[
                {"name": "Notion", "type": "app"},
                {"name": "Safari", "type": "app"},
                {"name": "Spotify", "type": "app"},
            ],
            sensory_mode="low_power",
        ),
        
        "research": WorkspaceProfile(
            name="Research",
            description="Research and learning workspace",
            trigger_phrases=["research session", "learning", "study", "reading"],
            apps=[
                {"name": "Safari", "type": "app"},
                {"name": "Notion", "type": "app"},
                {"name": "ChatGPT", "type": "app"},
                {"name": "Spotify", "type": "app"},
            ],
            sensory_mode="normal",
        ),
        
        "brainstorm": WorkspaceProfile(
            name="Brainstorming",
            description="Ideation and brainstorming workspace",
            trigger_phrases=["brainstorm", "brainstorming", "ideas", "ideation"],
            apps=[
                {"name": "Notion", "type": "app"},
                {"name": "Spotify", "type": "app"},
            ],
            sensory_mode="normal",
        ),
        
        "general": WorkspaceProfile(
            name="General",
            description="General purpose workspace",
            trigger_phrases=["general", "default", "basic", "just workspace"],
            apps=[
                {"name": "Notion", "type": "app"},
                {"name": "Terminal", "type": "app"},
                {"name": "Spotify", "type": "app"},
            ],
            sensory_mode="normal",
        ),
    }


class WorkspaceConfigManager:
    """Manages workspace configuration persistence."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or WORKSPACES_CONFIG
        self.profiles: Dict[str, WorkspaceProfile] = {}
        self.active_workspace: Optional[ActiveWorkspace] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create defaults."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Load profiles with validation
                for name, profile_data in data.get("profiles", {}).items():
                    try:
                        profile = WorkspaceProfile(**profile_data)
                        errors = profile.validate()
                        if errors:
                            log_debug(f"Profile '{name}' has validation errors: {errors}")
                        self.profiles[name] = profile
                    except TypeError as e:
                        log_debug(f"Failed to load profile '{name}': {e}")
                
                # Load active workspace if exists
                if data.get("active_workspace"):
                    try:
                        self.active_workspace = ActiveWorkspace.from_dict(
                            data["active_workspace"]
                        )
                    except (KeyError, TypeError, ValueError) as e:
                        log_debug(f"Failed to load active workspace: {e}")
                        self.active_workspace = None
                
                log_debug(f"Loaded {len(self.profiles)} workspace profiles")
                
                # If no valid profiles, use defaults
                if not self.profiles:
                    log_debug("No valid profiles loaded, using defaults")
                    self._use_defaults()
                
            except json.JSONDecodeError as e:
                log_debug(f"Config file is corrupted: {e}")
                # Backup corrupted file
                backup_path = self.config_path.with_suffix(f".json.backup.{int(datetime.now().timestamp())}")
                try:
                    self.config_path.rename(backup_path)
                    log_debug(f"Backed up corrupted config to {backup_path}")
                except Exception as be:
                    log_debug(f"Failed to backup corrupted config: {be}")
                self._use_defaults()
                self._save_config()
            except Exception as e:
                log_debug(f"Failed to load config: {e}")
                self._use_defaults()
        else:
            self._use_defaults()
            self._save_config()
    
    def _use_defaults(self):
        """Load default profiles."""
        self.profiles = get_default_profiles()
        log_debug("Using default workspace profiles")
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            data = {
                "profiles": {
                    name: asdict(profile) 
                    for name, profile in self.profiles.items()
                },
                "active_workspace": (
                    self.active_workspace.to_dict() 
                    if self.active_workspace else None
                ),
            }
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            log_debug(f"Failed to save config: {e}")
    
    def get_profile(self, name: str) -> Optional[WorkspaceProfile]:
        """Get a profile by name."""
        return self.profiles.get(name)
    
    def match_profile(self, user_input: str) -> Optional[WorkspaceProfile]:
        """
        Find a profile matching user input.
        
        FIX: Uses priority matching - longer trigger phrases match first
        to avoid "3d" matching before "3d modeling".
        """
        user_lower = user_input.lower()
        
        # Collect all matches with their longest matching phrase length
        matches: List[tuple] = []  # (phrase_length, profile)
        
        for profile in self.profiles.values():
            for phrase in profile.trigger_phrases:
                phrase_lower = phrase.lower()
                
                # Check for match (with word boundary for short phrases)
                if len(phrase_lower) <= 3:
                    pattern = rf'\b{re.escape(phrase_lower)}\b'
                    if re.search(pattern, user_lower):
                        matches.append((len(phrase_lower), profile))
                        break  # Found a match for this profile
                else:
                    if phrase_lower in user_lower:
                        matches.append((len(phrase_lower), profile))
                        break
        
        if matches:
            # Return the profile with the longest matching phrase
            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]
        
        return None
    
    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())
    
    def add_profile(self, profile: WorkspaceProfile) -> bool:
        """
        Add or update a profile.
        Returns True if successful, False if validation failed.
        """
        errors = profile.validate()
        if errors:
            log_debug(f"Profile validation failed: {errors}")
            return False
        
        key = profile.name.lower().replace(" ", "_")
        self.profiles[key] = profile
        self._save_config()
        return True
    
    def remove_profile(self, name: str) -> bool:
        """Remove a profile by name."""
        if name in self.profiles:
            del self.profiles[name]
            self._save_config()
            return True
        return False
    
    def set_active_workspace(self, workspace: ActiveWorkspace):
        """Set the currently active workspace."""
        self.active_workspace = workspace
        self._save_config()
    
    def clear_active_workspace(self):
        """Clear the active workspace."""
        self.active_workspace = None
        self._save_config()
    
    def get_active_workspace(self) -> Optional[ActiveWorkspace]:
        """Get the currently active workspace."""
        return self.active_workspace
    
    def reset_to_defaults(self):
        """Reset all profiles to defaults and clear active workspace."""
        self._use_defaults()
        self.active_workspace = None
        self._save_config()
        log_debug("Configuration reset to defaults")


# Singleton instance
_config_manager: Optional[WorkspaceConfigManager] = None

def get_config_manager() -> WorkspaceConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = WorkspaceConfigManager()
    return _config_manager


def reset_config_manager():
    """Reset the global config manager (useful for testing)."""
    global _config_manager
    _config_manager = None
