"""
ARGUS Robot Embodiment (STUB)
==============================

Future robot body for ARGUS.  Implements the same Embodiment interface
as DesktopEmbodiment, but with hardware I/O instead of desktop I/O.

When you're ready to build this:
    1. Implement each abstract method
    2. Register robot-specific capabilities (move_arm, navigate, look_at)
    3. Create main_robot.py entry point
    4. The brain stays identical — same BrainCore, same reasoning pipeline

The whole point of the architecture is that this file can exist
without changing a single line in brain/.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from embodiments.base import Embodiment


class RobotEmbodiment(Embodiment):
    """Robot body for ARGUS.  NOT YET IMPLEMENTED."""

    @property
    def name(self) -> str:
        return "robot"

    def start(self) -> None:
        raise NotImplementedError("Robot embodiment not yet built")

    def stop(self) -> None:
        raise NotImplementedError

    def get_user_input(self) -> Optional[str]:
        raise NotImplementedError

    def ask_user(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

    def deliver_response(self, text: str) -> None:
        raise NotImplementedError

    def show_debug(self, text: str) -> None:
        raise NotImplementedError

    def execute_platform_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_platform_capabilities(self) -> List[str]:
        return [
            # Future:
            # "robot.move_arm",
            # "robot.navigate",
            # "robot.look_at",
            # "robot.grasp",
        ]
