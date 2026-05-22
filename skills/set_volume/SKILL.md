---
name: set_volume
description: Control system audio volume.
version: 1.0.0
triggers: []
embodiments: []
parameters:
  command:
    type: string
    required: true
    description: "Natural language volume command."
---

Control system audio volume. Phrase the request the way the user said it ('volume up', 'mute', 'set volume to 50').
