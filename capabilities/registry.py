"""
Capability Registry — maps intent labels to actions.

Every action ARGUS can do is a Capability with a name, handler function,
and optional embodiment restrictions. The brain looks up capabilities
by intent label. The planner lists available capabilities for decomposition.

The brain asks: "what can I do with intent X on embodiment Y?" The registry answers.

Platform-agnostic capabilities (weather, stocks, calculator, web search) have no required_embodiments — they work everywhere.
Platform-specific capabilities (open_app, volume_control, move_arm) list which embodiment(s) they need.

Singleton: import REGISTRY from anywhere, it's the same instance.

Usage:
    from capabilities.registry import REGISTRY, Capability

    REGISTRY.register(Capability(
        name="weather.get_forecast",
        description="Get current or forecast weather for a city",
        handler=get_weather,
        keywords=["weather_data"],
        parameters={"city": "str", "day_offset": "int"},
    ))

    cap = REGISTRY.find_by_intent("weather_data")
    result = cap.handler(raw_text="weather in Syracuse", entities={})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Capability:
    """A thing ARGUS can do."""
    name: str                           # "weather.get_forecast", "desktop.open_app"
    description: str                    # shown to user if capability is unavailable. is human readable, not a code comment.
    handler: Callable[..., Any]         # the actual function — returns str, dict, or None
    parameters: Dict[str, str] = field(default_factory=dict)      # param → type hint
    required_embodiments: List[str] = field(default_factory=list)  # empty = works everywhere
    keywords: List[str] = field(default_factory=list)              # intent labels that map here
    needs_confirmation: bool = False    # planner will ask user before running (future)


class CapabilityRegistry:
    """
    Central registry of everything ARGUS can do.
    
    Two indexes:
        _caps:         name → Capability  (direct lookup by name)
        _intent_index: intent_label → name (used by brain._route)
    
    Embodiments register capabilities at startup.
    The brain queries by intent. The planner queries by embodiment.
    """
    def __init__(self) -> None:
        self._caps: Dict[str, Capability] = {}
        self._intent_index: Dict[str, str] = {}  # intent_label → cap_name

    #registration 
    def register(self, cap: Capability) -> None:
        """Register a capability. Overwrites if name already exists."""
        self._caps[cap.name] = cap
        for kw in cap.keywords:
            self._intent_index[kw] = cap.name

    def unregister(self, name: str) -> None:
        cap = self._caps.pop(name, None)
        if cap:
            for kw in cap.keywords:
                self._intent_index.pop(kw, None)

    #lookups 
    def get(self, name: str) -> Optional[Capability]:
        """Lookup by exact capability name."""
        return self._caps.get(name)

    def find_by_intent(self, intent_label: str) -> Optional[Capability]:
        """
        Map an intent classification label to a capability.
        Main lookup used by brain._route(). Intent classifier outputs labels
        like "weather_data" or "open" — this maps them to capabilities.
        Returns None if nobody handles this intent (brain falls through to LLM).
        """
        cap_name = self._intent_index.get(intent_label)
        if cap_name:
            return self._caps.get(cap_name)
        return None

    def list_all(self) -> List[Capability]:
        """Return all registered capabilities."""
        return list(self._caps.values())

    def list_for_embodiment(self, embodiment_name: str) -> List[Capability]:
        """
        Return capabilities available on a specific embodiment.

        What can this body do? 
        Filters out capabilities that require a different embodiment. 
        Used by the planner to know what tools to include when building multi-step plans.
        """
        return [
            c for c in self._caps.values()
            if not c.required_embodiments
            or embodiment_name in c.required_embodiments
        ]

    def list_names(self) -> List[str]:
        """Return all capability names (useful for world state)."""
        return list(self._caps.keys())

    def search(self, query: str) -> List[Capability]:
        """Fuzzy search across names, descriptions, and keywords."""
        q = query.lower()
        results = []
        for cap in self._caps.values():
            if (q in cap.name.lower()
                    or q in cap.description.lower()
                    or any(q in kw.lower() for kw in cap.keywords)):
                results.append(cap)
        return results

    def __len__(self) -> int:
        return len(self._caps)

    def __repr__(self) -> str:
        return f"<CapabilityRegistry caps={len(self._caps)}>"


# ── module-level singleton ──
REGISTRY = CapabilityRegistry()
