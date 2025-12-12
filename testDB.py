import sqlite3
from pathlib import Path

db = Path("option_bot_spreads/data/ml_training_data.db")
conn = sqlite3.connect(db)
cursor = conn.execute("SELECT COUNT(*) FROM ml_spread_candidates")
print("Rows in ml_spread_candidates:", cursor.fetchone()[0])
conn.close()
