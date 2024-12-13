from fastapi import FastAPI, HTTPException, Query
from api.database import get_connection, initialize_database

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Initialize the database when the application starts."""
    initialize_database()

@app.get("/items")
def get_items(limit: int = Query(default=10, ge=1, le=100)):
    """
    Retrieve a list of items from the database.
    - `limit`: The maximum number of items to return (default: 10).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items LIMIT ?", (limit,))
    items = cursor.fetchall()
    conn.close()
    return [dict(item) for item in items]

@app.post("/items")
def create_item(name: str, value: float):
    """
    Create a new item in the database.
    - `name`: Name of the item.
    - `value`: Value associated with the item.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO items (name, value) VALUES (?, ?)", (name, value))
    conn.commit()
    item_id = cursor.lastrowid
    conn.close()
    return {"id": item_id, "name": name, "value": value}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    """
    Retrieve a specific item by its ID.
    - `item_id`: The ID of the item to retrieve.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items WHERE id = ?", (item_id,))
    item = cursor.fetchone()
    conn.close()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return dict(item)

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """
    Delete a specific item by its ID.
    - `item_id`: The ID of the item to delete.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
    changes = conn.total_changes
    conn.commit()
    conn.close()
    if changes == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"detail": "Item deleted successfully"}
