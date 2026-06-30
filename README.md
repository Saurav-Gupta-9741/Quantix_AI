# Quantix AI 

An institutional-grade AI trading terminal powered by an ensemble of Deep Learning models.

## Architecture

- **Backend:** FastAPI (Python)
- **Database & Concurrency:** SQLite in WAL Mode. The system uses a Dual-Layer Concurrency Model:
  1. *Intra-Process:* `threading.RLock()` protects against races between FastAPI async worker threads.
  2. *Inter-Process:* SQLite WAL mode allows concurrent reads/writes between the main API process and the independent Risk Manager daemon.
- **Frontend:** Vanilla JS, CSS Variables, Server-Sent Events (SSE)
- **Containerization:** Docker & Docker Compose

## AI Models
1. **Time-Series Engine:** BiLSTM with Attention (predicts close price using historical technicals).
2. **Vision Engine:** CNN (ResNet-based) with Grad-CAM mapping for chart pattern recognition.
3. **Sentiment Engine:** FinBERT NLP for processing real-time news headlines.

## Features
- **Live Market Data:** Fetches up-to-the-minute data via `yfinance`.
- **Paper Trading:** Background worker automatically monitors watchlists and executes paper trades based on AI confidence.
- **Risk Management:** Position sizing via Kelly Criterion, and an automated background Take-Profit/Stop-Loss daemon.
- **Model Drift Detection:** Continually validates LSTM prediction accuracy against recent actuals and suppresses the model if it drifts.
- **Explainable AI:** Outputs Grad-CAM heatmaps directly in the UI to explain CNN decisions.

## Quick Start

### 1. Run via Docker (Recommended)
```bash
docker-compose up --build
```

### 2. Run Locally
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Start the API (will automatically launch the risk daemon):
```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000
```
3. Open `http://localhost:8000` in your browser.

## Project Structure
- `main_api.py`: FastAPI entrypoint, routing, ensemble logic.
- `shared_state.py`: SQLite `DatabaseManager` and connection pooling.
- `risk_manager.py`: Background thread/process monitoring open positions.
- `paper_trader.py`: Script to batch process watchlists and generate signals.
- `cnn_model.py`, `lstm_model.py`: PyTorch architectures.
- `index.html`: Web interface.
