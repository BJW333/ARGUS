---
name: get_crypto_price
description: Get the current price of a cryptocurrency.
version: 1.0.0
triggers: []
embodiments: []
parameters:
  coin:
    type: string
    required: true
    description: "Cryptocurrency name lowercase (e.g. 'bitcoin', 'ethereum', 'solana')."
---

Get the current price of a cryptocurrency. Use when the user asks about Bitcoin, Ethereum, Solana, or any other crypto.
