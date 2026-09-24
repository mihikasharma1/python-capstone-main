"""Create a synthetic SQLite database of customers and orders for the quantitative agent."""
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
random.seed(42)  # reproducible synthetic data


def quarter_of(d: date) -> str:
    return f"Q{(d.month - 1) // 3 + 1}"


def build():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'churned'))
        )
    """)
    cur.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date TEXT NOT NULL,
            amount REAL NOT NULL,
            region TEXT NOT NULL,
            year INTEGER NOT NULL,
            quarter TEXT NOT NULL
        )
    """)

    customers = []
    for cid in range(1, 41):
        region = random.choice(REGIONS)
        signup = date(2023, 1, 1) + timedelta(days=random.randint(0, 700))
        status = random.choices(["active", "churned"], weights=[0.8, 0.2])[0]
        customers.append((cid, f"Customer {cid}", region, signup.isoformat(), status))
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    orders = []
    order_id = 1
    for cid, name, region, signup_str, status in customers:
        signup = date.fromisoformat(signup_str)
        for _ in range(random.randint(3, 15)):
            order_date = signup + timedelta(days=random.randint(0, 600))
            if order_date > date(2024, 12, 31):
                continue
            amount = round(random.uniform(50, 2500), 2)
            orders.append((
                order_id, cid, order_date.isoformat(), amount,
                region, order_date.year, quarter_of(order_date),
            ))
            order_id += 1
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"Seeded {len(customers)} customers and {len(orders)} orders into {DB_PATH}")


if __name__ == "__main__":
    build()