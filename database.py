import sqlite3
import time
from typing import List, Tuple


class Database:
    def __init__(self, path: str = "chat.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                ts INTEGER
            )
            """
        )
        self.conn.commit()

    def save_message(self, role: str, content: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (role, content, ts) VALUES (?, ?, ?)",
            (role, content, int(time.time())),
        )
        self.conn.commit()

    def get_recent_messages(self, limit: int = 30) -> List[Tuple[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        # return in chronological order (oldest first)
        return list(reversed(rows))

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
