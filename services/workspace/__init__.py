"""
ARGUS Workspace Package
=======================

Voice-driven project workspace manager for ARGUS.
Automatically launches the right tools for different types of work.

Usage:
    from workspace import ArgusWorkspaceIntegration
    
    workspace = ArgusWorkspaceIntegration(
        speak_fn=speak,
        listen_fn=generalvoiceinput,
    )
"""

from .workspace_config import (
    WorkspaceProfile,
    ActiveWorkspace,
    WorkspaceConfigManager,
    get_config_manager,
    WORKSPACE_ROOT,
    PROJECTS_ROOT,
)

from .workspace_manager import (
    WorkspaceManager,
    create_workspace_manager,
)

from .workspace_integration import (
    ArgusWorkspaceIntegration,
    quick_start_workspace,
)

__all__ = [
    # Config
    "WorkspaceProfile",
    "ActiveWorkspace", 
    "WorkspaceConfigManager",
    "get_config_manager",
    "WORKSPACE_ROOT",
    "PROJECTS_ROOT",
    # Manager
    "WorkspaceManager",
    "create_workspace_manager",
    # Integration
    "ArgusWorkspaceIntegration",
    "quick_start_workspace",
]
