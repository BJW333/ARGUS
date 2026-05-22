"""
ARGUS Brain Intent Classifier
==============================

Thin wrapper around nlp/intent.py for the brain layer.
Currently re-exports the existing intentrecognition class.

The original nlp/intent.py uses print_to_gui for 3 debug lines.
Those are non-critical — the classifier works without them.

Future: migrate intent.py logic fully into this module,
strip all platform imports, add new intent labels.
"""
from __future__ import annotations

# Re-export the existing class — brain/core.py imports from here
from brain.nlp.intent import intentrecognition as IntentClassifier

__all__ = ["IntentClassifier"]
