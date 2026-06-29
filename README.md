# OmniAsset Trading Engine: System Architecture & Feature Specification (v2.0)

This document provides a comprehensive overview of the Quantix AI Terminal architecture, its subsystems, and execution flows. It has been fully updated to reflect the enterprise-grade architectural hardening (SQLite WAL, SSE Streams, Model Registries, and Circuit Breakers) implemented during the latest LLM audit.

---

## 1. Executive Summary

Quantix AI is a localized, Human-in-the-Loop institutional trading terminal. It acts as an overarching orchestrator that combines three independent Deep Learning paradigms (Time-Series, Computer Vision, and Natural Language Processing) into a unified execution engine. Rather than relying on a single indicator, Quantix AI requires **consensus** across all neural networks before recommending a trade, which is then dynamically sized via a Reinforcement Learning manager.

---

## 2. Core Architecture & Stack

### Frontend UI
- **Technology:** Pure HTML5, CSS3, Vanilla JavaScript.
- **Library:** `Lightweight-Charts` (by TradingView) for rendering interactive candlestick backtest graphs.
- **Paradigm:** Single Page Application (SPA).
- **Communication:** Standard HTTP REST for commands, and **Server-Sent Events (SSE)** via `EventSource` for real-time, zero-latency portfolio updates pushed by background daemons.

### Backend Infrastructure
- **Server:** Python `FastAPI` running on a `uvicorn` ASGI server.
- **Database:** `SQLite3` operating in **WAL (Write-Ahead Logging)** mode. This entirely prevents race conditions by allowing multiple disjoint processes (e.g., API server, Risk Manager daemon, Paper Trader) to read and modify the database concurrently with row-level atomic locks.
- **Data Pipeline:** Direct connections to `yfinance` for up-to-the-minute ticking data, fortified by a `CircuitBreaker` utility that prevents cascading failures if Yahoo Finance enforces rate-limits.

---

## 3. The 4-Pillar AI Ensemble

The system avoids the single-model failure trap by isolating four distinct AIs. A trade signal is generated only if the overarching logic engine determines a high confidence consensus.

### Pillar 1: Time-Series Forecasting (LSTM + Attention)
- **Model:** Bi-directional Long Short-Term Memory (BiLSTM) network with a custom Self-Attention layer.
- **Function:** Analyzes the last 30 days of mathematical market data (Close Price, Volume, RSI, MACD, MACD Signal).
- **Output:** Predicts tomorrow's exact closing price.
- **Safety Mechanism:** Protected by `check_model_drift()`. Before executing, the model tests itself against the last 15 days of actual prices. If the Root Mean Square Error (RMSE) exceeds 5%, the model is flagged as "Drifting" (stale weights) and its prediction is suppressed.

### Pillar 2: Visual Pattern Recognition (CNN)
- **Model:** PyTorch ResNet-18 Convolutional Neural Network.
- **Function:** Instead of looking at numbers, it physically "looks" at a chart. It takes the mathematical sequence, draws a candlestick chart in-memory using `FigureCanvasAgg` (ensuring 100% thread safety), converts it to an RGB tensor, and uses computer vision to detect shapes.
- **Output:** Classifies the chart into: Bull Flag, Bear Flag, Head & Shoulders, Double Bottom, or Consolidation.

### Pillar 3: Sentiment & Fundamental Analysis (FinBERT / RAG)
- **Model:** FinBERT (Transformers).
- **Function:** Uses a Retrieval-Augmented Generation (RAG) architecture to scrape live news headlines for the specific asset.
- **Output:** Calculates a polarized sentiment score (Fear vs. Greed). The LLM Advisor then synthesizes this raw numerical sentiment into a human-readable strategic briefing.

### Pillar 4: Risk & Position Sizing (RL Agent)
- **Model:** Kelly Criterion Allocator.
- **Function:** Calculates the mathematical optimal position size based on historical asset volatility and the ensemble's confidence score.
- **Execution:** Uses an institutional "Half-Kelly" safety factor, dynamically clipping exposure between a 3% absolute floor and a 15% maximum portfolio limit per trade.

---

## 4. Execution Workflow

### A. The Research Dashboard (Manual Query)
1. User searches a global ticker (e.g., `NVDA`, `BTC-USD`, `RELIANCE.NS`).
2. The API triggers a `yfinance` fetch. (If yfinance fails 5 consecutive times, the `CircuitBreaker` opens for 5 minutes, gracefully degrading the UI).
3. The data is pre-processed and fed simultaneously into the frozen `ModelRegistry` tensors (wrapped in `torch.inference_mode()` to prevent gradient memory leaks).
4. The system calculates an overarching Confidence Score and outputs a BUY, SELL, or HOLD recommendation with expected ROI and precise position sizing.

### B. The Paper Trading Sweep (Daemon Automation)
1. User clicks **"▶ SWEEP MARKET FOR RECOMMENDATIONS"**.
2. A background `paper_trader.py` thread iterates through the hardcoded institutional `WATCHLIST`.
3. It performs the complete Deep Learning analysis on every stock in the background.
4. Any stock that achieves a "BUY" consensus is appended to the SQLite `signals` table as a Pending Trade.
5. The UI automatically renders these pending trades, awaiting human-in-the-loop approval to physically deduct virtual cash and move the asset into `holdings`.

### C. Auto-Take-Profit Risk Manager (Headless Daemon)
1. When the `uvicorn` server starts, it spawns `risk_manager.py` as an invisible, headless background daemon.
2. Every 5 minutes, this daemon queries the SQLite `holdings` table.
3. It compares the `entry_price` of your active assets against the live ticking price.
4. **The Trigger:** If any holding crosses the `+5.0%` Profit threshold, the daemon executes a ruthless algorithmic market sell.
5. **The Stream:** Upon selling, the daemon injects a `{"type": "refresh"}` payload into the internal ASGI Event Queue.
6. The frontend `EventSource` receives this Server-Sent Event (SSE) and visually redraws your PnL on the screen in real-time.
