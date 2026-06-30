import sqlite3
import threading
import os
from contextlib import contextmanager

# Dual-Layer Concurrency Model:
# 1. Intra-Process: _db_lock (threading.RLock) protects against race conditions between FastAPI worker threads.
# 2. Inter-Process: SQLite WAL (Write-Ahead Logging) mode enables concurrent reads/writes between the main FastAPI process and the background risk_manager.py subprocess.
_db_lock = threading.RLock()
DB_PATH = os.path.join(os.path.dirname(__file__), "trading.db")

class DatabaseManager:
    """Manages SQLite connections with RLock (for threads) and WAL mode (for processes)."""
    
    @contextmanager
    def get_read_connection(self, timeout_val=5.0):
        """Yields a read-only connection without acquiring the RLock, maximizing WAL concurrency."""
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=timeout_val)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_write_connection(self, timeout_val=5.0):
        """Yields a write connection, acquiring the RLock to serialize intra-process write atomicity."""
        with _db_lock:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=timeout_val)
            try:
                yield conn
            finally:
                conn.close()

db = DatabaseManager()

def init_db():
    with db.get_write_connection() as conn:
        # PROOF: WAL mode is a persistent database setting. Executed once on init, not per-connection.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY,
                balance_usd REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                ticker TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                total_cost REAL NOT NULL,
                date TEXT NOT NULL,
                currency TEXT DEFAULT 'USD',
                stop_loss_pct REAL DEFAULT -3.0,
                take_profit_pct REAL DEFAULT 5.0
            )
        """)
        # Migrations for holdings
        try:
            conn.execute("ALTER TABLE holdings ADD COLUMN stop_loss_pct REAL DEFAULT -3.0")
        except sqlite3.OperationalError: pass
        try:
            conn.execute("ALTER TABLE holdings ADD COLUMN take_profit_pct REAL DEFAULT 5.0")
        except sqlite3.OperationalError: pass
        try:
            conn.execute("ALTER TABLE holdings ADD COLUMN currency TEXT DEFAULT 'USD'")
        except sqlite3.OperationalError: pass
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                ticker TEXT PRIMARY KEY,
                signal TEXT,
                confidence TEXT,
                predicted_roi TEXT,
                current_price REAL,
                recommended_qty REAL,
                cost REAL,
                date TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_msg TEXT NOT NULL,
                date TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker TEXT PRIMARY KEY
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_equity REAL NOT NULL
            )
        """)
        
        # Seed default watchlist if empty
        cursor = conn.execute("SELECT COUNT(*) FROM watchlist")
        if cursor.fetchone()[0] == 0:
            defaults = [
                "AAPL", "MSFT", "NVDA", "TSLA",
                "BTC-USD", "ETH-USD",
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
                "XOM", "CVX"
            ]
            conn.executemany("INSERT INTO watchlist (ticker) VALUES (?)", [(t,) for t in defaults])
        
        # Ensure portfolio row exists
        cursor = conn.execute("SELECT id FROM portfolio WHERE id=1")
        if not cursor.fetchone():
            conn.execute("INSERT INTO portfolio (id, balance_usd) VALUES (1, 100000.0)")
            # Add initial equity snapshot
            import datetime
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute("INSERT INTO equity_snapshots (timestamp, total_equity) VALUES (?, ?)", (now_str, 100000.0))
            
        conn.commit()


def get_balance():
    with db.get_read_connection() as conn:
        cursor = conn.execute("SELECT balance_usd FROM portfolio WHERE id=1")
        row = cursor.fetchone()
        return row[0] if row else 100000.0

def get_holdings():
    with db.get_read_connection() as conn:
        cursor = conn.execute("SELECT * FROM holdings")
        return cursor.fetchall()

def update_balance(new_balance):
    with db.get_write_connection() as conn:
        conn.execute("UPDATE portfolio SET balance_usd = ? WHERE id=1", (new_balance,))
        conn.commit()
