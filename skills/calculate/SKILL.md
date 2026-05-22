---
name: calculate
description: Evaluate a math expression.
version: 1.0.0
triggers: []
embodiments: []
parameters:
  expression:
    type: string
    required: true
    description: "Math expression as text."
---

Evaluate a math expression. Use for any arithmetic the user asks. Supports +, -, *, /, **, %, parentheses, and natural-language phrases like 'three times five'.
