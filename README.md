---
title: StockPilot Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# StockPilot Agent

StockPilot Agent turns a simple stock-news sentiment analyzer into a personal research workflow for long-term investors who cannot watch the market all day.

## What It Does

1. Parses the user's analysis goal.
2. Collects recent stock headlines from FinViz.
3. Scores news sentiment with NLTK VADER.
4. Detects risk signals across regulation, earnings, competition, macro and price movement.
5. Generates a structured memo that can be saved into an investment journal or used to refresh a personal watchlist.

## Why It Feels Agentic

Instead of only showing a sentiment chart, the app shows a multi-step workflow:

- Briefing Agent: understands the user's goal.
- News Scout Agent: collects and prepares evidence.
- Sentiment Agent: scores each headline.
- Risk Agent: extracts warning signals.
- Portfolio Copilot: writes the final decision memo.

The Briefing Agent and Portfolio Copilot can use an LLM when `OPENAI_API_KEY` is configured. If no key is available, the app automatically falls back to the rule-based workflow.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional LLM mode:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```

## Usage Tips

For best results, use US stock tickers with active news coverage.

The analysis goal field changes the workflow focus. Examples:

- Identify long-term holding risks.
- Scan for positive growth catalysts.
- Review earnings and guidance signals.
- Watch for regulation, lawsuits or compliance issues.
- Compare recent news with an existing holding thesis.

The app can optionally use offline sample data when live scraping fails. Offline sample data is clearly labeled and should not be treated as current market information.
