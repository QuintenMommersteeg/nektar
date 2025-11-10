import sqlite3
from contextlib import contextmanager

DB_PATH = "demo.db"

@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()

def get_customers():
    with db() as con:
        cur = con.execute("SELECT id, name, email FROM customers ORDER BY name ASC")
        return [dict(r) for r in cur.fetchall()]

def get_customer_by_name(name: str):
    with db() as con:
        cur = con.execute(
            "SELECT id, name, email FROM customers WHERE name LIKE ? LIMIT 1",
            (f"%{name}%",),
        )
        row = cur.fetchone()
        return dict(row) if row else None

def get_recent_work_orders(limit: int = 3):
    with db() as con:
        cur = con.execute(
            """
            SELECT w.id, w.title, w.description, w.status, w.created_at, c.name as customer_name
            FROM work_orders w
            LEFT JOIN customers c ON c.id = w.customer_id
            ORDER BY datetime(w.created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

def get_work_orders_for_customer(customer_name: str):
    with db() as con:
        cur = con.execute(
            """
            SELECT w.id, w.title, w.description, w.status, w.created_at, c.name as customer_name
            FROM work_orders w
            LEFT JOIN customers c ON c.id = w.customer_id
            WHERE c.name LIKE ?
            ORDER BY datetime(w.created_at) DESC
            """,
            (f"%{customer_name}%",),
        )
        return [dict(r) for r in cur.fetchall()]

def get_next_appointment_for_customer(customer_name: str):
    with db() as con:
        cur = con.execute(
            """
            SELECT a.id, a.start_ts, a.end_ts, a.resource, c.name as customer_name
            FROM appointments a
            LEFT JOIN customers c ON c.id = a.customer_id
            WHERE c.name LIKE ?
            ORDER BY datetime(a.start_ts) ASC
            LIMIT 1
            """,
            (f"%{customer_name}%",),
        )
        row = cur.fetchone()
        return dict(row) if row else None

# ===== Datumfilters =====

def get_work_orders_on_date(date: str):
    """date = 'YYYY-MM-DD' (matcht op created_at-datum)."""
    with db() as con:
        cur = con.execute(
            """
            SELECT w.id, w.title, w.description, w.status, w.created_at, c.name as customer_name
            FROM work_orders w
            LEFT JOIN customers c ON c.id = w.customer_id
            WHERE substr(w.created_at,1,10) = ?
            ORDER BY datetime(w.created_at) ASC
            """,
            (date,),
        )
        return [dict(r) for r in cur.fetchall()]

def get_appointments_on_date(date: str):
    """date = 'YYYY-MM-DD' (matcht op start_ts-datum)."""
    with db() as con:
        cur = con.execute(
            """
            SELECT a.id, a.start_ts, a.end_ts, a.resource, c.name as customer_name
            FROM appointments a
            LEFT JOIN customers c ON c.id = a.customer_id
            WHERE substr(a.start_ts,1,10) = ?
            ORDER BY datetime(a.start_ts) ASC
            """,
            (date,),
        )
        return [dict(r) for r in cur.fetchall()]

# ===== Vrij zoeken (vage vragen opvangen) =====

def search_all(q: str):
    """Zoek breed in werkorders (titel/omschrijving/klant) en afspraken (klant/resource)."""
    like = f"%{q}%"
    with db() as con:
        wcur = con.execute(
            """
            SELECT w.id, w.title, w.description, w.status, w.created_at, c.name as customer_name
            FROM work_orders w
            LEFT JOIN customers c ON c.id = w.customer_id
            WHERE w.title LIKE ? OR w.description LIKE ? OR c.name LIKE ?
            ORDER BY datetime(w.created_at) DESC
            LIMIT 20
            """,
            (like, like, like),
        )
        work = [dict(r) for r in wcur.fetchall()]

        acur = con.execute(
            """
            SELECT a.id, a.start_ts, a.end_ts, a.resource, c.name as customer_name
            FROM appointments a
            LEFT JOIN customers c ON c.id = a.customer_id
            WHERE c.name LIKE ? OR a.resource LIKE ?
            ORDER BY datetime(a.start_ts) DESC
            LIMIT 20
            """,
            (like, like),
        )
        appt = [dict(r) for r in acur.fetchall()]

    return {"query": q, "work_orders": work, "appointments": appt}