---
name: get_weather
description: Get the CURRENT weather or forecast for a specific city.
version: 1.0.0
triggers: []
embodiments: []
parameters:
  city:
    type: string
    required: true
    description: "Specific CITY name with state or country (e.g. 'Red Bank, NJ', 'Syracuse, NY', 'London, UK'). NEVER pass just a state ('New Jersey') or country ('USA') — the API needs a specific city. If the user only gives a state, call ask_user to get the city."
  day_offset:
    type: integer
    required: false
    default: 0
    description: "0 for today, 1 for tomorrow, up to 6 for a week out."
---

Get the CURRENT weather or forecast for a specific city. ALWAYS call this for any weather-related question — never estimate from prior conversation or memory. Memory may contain stale weather from previous days. This tool returns live data from the weather API. Use it for: current temps, forecasts, rain/snow checks, jacket/umbrella questions, outfit suggestions based on weather, anything weather-adjacent.
