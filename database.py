import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / 'laptop_performance.db'
# Now DB_path will be the location of database file in hard drive 


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu REAL,
                ram REAL,
                disk REAL,
                temperature REAL,
                battery REAL,
                network_mb REAL,
                processes INTEGER,
                uptime_hours REAL,
                plugged INTEGER
            )
        ''')
        conn.commit()


def insert_metric(data, timestamp=None):
    with get_connection() as conn:
        if timestamp is None:
            conn.execute('''
                INSERT INTO performance_log
                (timestamp, cpu, ram, disk, temperature, battery,
                 network_mb, processes, uptime_hours, plugged)
                VALUES (datetime('now','localtime'),?,?,?,?,?,?,?,?,?)
            ''', (
                data['cpu'], data['ram'], data['disk'], data['temperature'],
                data['battery'], data['network_mb'], data['processes'],
                data['uptime_hours'], int(data['plugged'])
            ))
        else:
            conn.execute('''
                INSERT INTO performance_log
                (timestamp, cpu, ram, disk, temperature, battery,
                 network_mb, processes, uptime_hours, plugged)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', (
                timestamp, data['cpu'], data['ram'], data['disk'], data['temperature'],
                data['battery'], data['network_mb'], data['processes'],
                data['uptime_hours'], int(data['plugged'])
            ))
        conn.commit()


def load_metrics(limit=1000):
    limit = max(1, int(limit))
    with get_connection() as conn:
        return pd.read_sql_query(
            f'SELECT * FROM performance_log ORDER BY id DESC LIMIT {limit}', conn
        ).sort_values('id').reset_index(drop=True)


def count_metrics():
    with get_connection() as conn:
        return int(conn.execute('SELECT COUNT(*) FROM performance_log').fetchone()[0])


def clear_metrics():
    with get_connection() as conn:
        conn.execute('DELETE FROM performance_log')
        conn.commit()
