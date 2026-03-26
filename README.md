# Financial Investor Note Generator

An AI-powered multi-agent system that parses equity questions such as "Should I buy NVIDIA?", gathers market and filing data, computes core metrics, and produces a polished investor note.

All agent outputs are passed within the active session only; the current design does not rely on persistent shared storage.

## Overview

This tool streamlines the equity research workflow by:

1. **Parsing the User Prompt** - Detects that the user is asking for an equity opinion and extracts the company or ticker for downstream processing.
2. **Retrieving Market Data** - Pulls stock price and related market data from `yfinance`.
3. **Analyzing Recent Filings** - Retrieves the past four 10-Q filings and runs sentiment or tone analysis on the filing text.
4. **Computing Core Metrics** - Calculates basic valuation and performance metrics such as P/E and EBITDA.
5. **Drafting and Refining the Report** - Produces an initial write-up with visualizations, then performs a final senior-analyst editing pass tailored to the user's requirements.

## Multi-Agent Workflow

The current architecture uses five coordinated agents:

1. **Mavis** - Queries `yfinance` for stock prices and related market data.
2. **Henry + Leran** - Retrieve the past four quarterly 10-Q filings and analyze the tone of those filings.
3. **Financial Calculation Agent** - Combines filing and market inputs to compute metrics such as P/E and EBITDA.
4. **Zipin** - Creates the main report draft and generates visualizations.
5. **Jason** - Edits the draft as a senior analyst, ensuring the final answer meets user-specific requirements.

## Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Backend | Python |
| Frontend | Node.js |
| Data sources | SEC EDGAR, yfinance |
| Output | Markdown, PDF, or UI report |

## Project Structure

```
.
├── backend/        # Python backend for orchestration, retrieval, analysis, and reporting
├── frontend/       # Node.js frontend for prompt input and report display
├── architecture    # High-level multi-agent architecture diagram
└── README.md
```

## Getting Started

_Setup instructions will be added as the project develops._

## Team

DSA4265 Group 8
