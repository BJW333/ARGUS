---
name: get_stock_price
description: Get the latest price for a stock.
version: 1.0.0
triggers: []
embodiments: []
parameters:
  company:
    type: string
    required: true
    description: "Company name or ticker symbol (e.g. 'Apple', 'AAPL', 'Tesla')."
---

Get the latest price for a stock. Use when the user asks about a company's share price, market price, or stock performance.
