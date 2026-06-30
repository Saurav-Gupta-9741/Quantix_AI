import threading
import time
import random
import sqlite3
from shared_state import db, DB_PATH, _db_lock
from collections import defaultdict

# Instrumented Load Test for SQLite Concurrency (Fix 15)
# Records exact wait distribution and forces a deliberate failure path.

stats = {
    "successful_reads": 0, 
    "successful_writes": 0, 
    "lock_timeouts": 0
}
wait_times = []
lock = threading.Lock()

def simulated_analysis_request(timeout_val):
    try:
        start_t = time.time()
        with db.get_read_connection(timeout_val=timeout_val) as conn:
            cursor = conn.execute("SELECT * FROM holdings")
            cursor.fetchall()
            acquired_t = time.time() # Lock acquisition time measured here
            
            # Simulate CPU work holding the connection open
            time.sleep(random.uniform(0.01, 0.05))
        
        wait = acquired_t - start_t
        with lock:
            stats["successful_reads"] += 1
            wait_times.append(wait)
    except Exception as e:
        if "locked" in str(e):
            with lock:
                stats["lock_timeouts"] += 1
        else:
            print(f"Read Error: {e}")

def simulated_risk_manager(timeout_val, simulate_inter_process=False):
    try:
        start_t = time.time()
        
        if simulate_inter_process:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=timeout_val)
        else:
            # We can't manually enter/exit a generator-based context manager easily
            # We will just acquire the lock manually for Phase 1
            _db_lock.acquire()
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=timeout_val)
            
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR REPLACE INTO holdings (ticker, qty, entry_price, total_cost, date) VALUES (?, ?, ?, ?, ?)", 
                         ("TEST", 1, 100, 100, "2026-06-30"))
            
            # Simulate API network delay while holding write lock
            time.sleep(0.1)
            
            conn.execute("DELETE FROM holdings WHERE ticker='TEST'")
            conn.commit()
            acquired_t = time.time() # Write lock acquisition/commit time measured here
        finally:
            conn.close()
            if not simulate_inter_process:
                _db_lock.release()
        
        wait = acquired_t - start_t - 0.1 # subtract the artificial API delay
        with lock:
            stats["successful_writes"] += 1
            wait_times.append(wait)
    except Exception as e:
        if "locked" in str(e):
            with lock:
                stats["lock_timeouts"] += 1
        else:
            print(f"Write Error: {e}")

def run_test_phase(phase_name, num_readers, num_writers, timeout_val, simulate_inter_process=False):
    print(f"\n[+] Starting {phase_name} (timeout={timeout_val}s)")
    global stats, wait_times
    stats = {"successful_reads": 0, "successful_writes": 0, "lock_timeouts": 0}
    wait_times = []
    
    threads = []
    for i in range(num_readers):
        threads.append(threading.Thread(target=simulated_analysis_request, args=(timeout_val,)))
    for i in range(num_writers):
        threads.append(threading.Thread(target=simulated_risk_manager, args=(timeout_val, simulate_inter_process)))
        
    random.shuffle(threads)
    
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    
    # Categorize wait times
    buckets = defaultdict(int)
    max_wait = 0
    for w in wait_times:
        max_wait = max(max_wait, w)
        if w < 0.05: buckets["<50ms"] += 1
        elif w < 0.2: buckets["50ms-200ms"] += 1
        elif w < 1.0: buckets["200ms-1s"] += 1
        else: buckets[">1s"] += 1
        
    print(f"[====== {phase_name} RESULTS ======]")
    print(f"Total Threads: {num_readers + num_writers} | Elapsed: {end_time - start_time:.2f}s")
    print(f"Successful Reads: {stats['successful_reads']}/{num_readers}")
    print(f"Successful Writes: {stats['successful_writes']}/{num_writers}")
    print(f"Lock Timeouts (Failed): {stats['lock_timeouts']}")
    print(f"Wait Time Distribution: {dict(buckets)}")
    print(f"Maximum Wait Time Observed: {max_wait:.3f}s")

if __name__ == "__main__":
    import os
    if not os.path.exists(DB_PATH):
        # Initialize DB briefly if it doesn't exist
        from shared_state import init_db
        init_db()

    # PHASE 1: Normal Operation (timeout 5.0)
    run_test_phase("PHASE 1: Standard Load", num_readers=50, num_writers=5, timeout_val=5.0)
    
    # PHASE 2: Forced Failure Path (timeout 0.05)
    # Proves the retry limit is hit and exceptions are successfully raised.
    run_test_phase("PHASE 2: Forced Failure Path", num_readers=50, num_writers=5, timeout_val=0.05, simulate_inter_process=True)
