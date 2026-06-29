import sqlite3
import threading

_db_lock = threading.Lock()
DB_PATH = "trading.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with _db_lock:
        conn = get_conn()
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
                date TEXT NOT NULL
            )
        """)
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
        
        # Ensure portfolio row exists
        cursor = conn.execute("SELECT id FROM portfolio WHERE id=1")
        if not cursor.fetchone():
            conn.execute("INSERT INTO portfolio (id, balance_usd) VALUES (1, 100000.0)")
            
        conn.commit()
        conn.close()

def get_balance():
    with _db_lock:
        conn = get_conn()
        cursor = conn.execute("SELECT balance_usd FROM portfolio WHERE id=1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 100000.0

def update_balance(new_balance):
    with _db_lock:
        conn = get_conn()
        conn.execute("UPDATE portfolio SET balance_usd = ? WHERE id=1", (new_balance,))
        conn.commit()
        conn.close()
