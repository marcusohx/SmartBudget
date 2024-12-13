import sqlite3

DATABASE_PATH = "./performance.db"

def get_connection():
    """Create and return a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enables dictionary-like access for rows
    return conn

def initialize_database():
    """Initialize the database and create necessary tables."""
    conn = get_connection()
    cursor = conn.cursor()
    # Example: Create a table for storing items
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        value REAL NOT NULL
    )
    """)
    conn.commit()
    conn.close()
