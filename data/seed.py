"""Generate the sample e-commerce SQLite database.

Dates are generated relative to *today*, so questions like
"last month" or "this quarter" always have data behind them,
no matter when the app is run or graded.

Run directly:  python -m data.seed  (writes data/sample.db)
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "sample.db"

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

# (name, category, price, base_return_probability)
PRODUCTS = [
    ("Trailblazer Running Shoes", "Footwear", 119.00, 0.22),
    ("Everyday Canvas Sneakers", "Footwear", 59.00, 0.15),
    ("Summit Hiking Boots", "Footwear", 169.00, 0.18),
    ("Cloudstep Slippers", "Footwear", 29.00, 0.06),
    ("Aero Wireless Earbuds", "Electronics", 89.00, 0.12),
    ("Pulse Fitness Tracker", "Electronics", 129.00, 0.14),
    ("Nova 4K Webcam", "Electronics", 99.00, 0.09),
    ("Volt Power Bank 20K", "Electronics", 45.00, 0.04),
    ("Echo Bluetooth Speaker", "Electronics", 79.00, 0.07),
    ("Merino Base Layer", "Apparel", 85.00, 0.11),
    ("Storm Shell Jacket", "Apparel", 199.00, 0.16),
    ("Classic Cotton Tee", "Apparel", 22.00, 0.08),
    ("Flex Yoga Leggings", "Apparel", 64.00, 0.19),
    ("Alpine Fleece Hoodie", "Apparel", 72.00, 0.10),
    ("Brew Master Pour-Over Set", "Home & Kitchen", 49.00, 0.05),
    ("Chef's Cast Iron Skillet", "Home & Kitchen", 39.00, 0.03),
    ("Aroma Smart Diffuser", "Home & Kitchen", 54.00, 0.13),
    ("Linen Throw Blanket", "Home & Kitchen", 68.00, 0.09),
    ("Ridge 30L Daypack", "Outdoor", 95.00, 0.07),
    ("Basecamp 2P Tent", "Outdoor", 249.00, 0.12),
    ("Hydro Insulated Bottle", "Outdoor", 32.00, 0.03),
    ("Trek Carbon Poles", "Outdoor", 139.00, 0.08),
    ("Glow Vitamin C Serum", "Beauty", 38.00, 0.10),
    ("Pure Mineral Sunscreen", "Beauty", 24.00, 0.05),
]

FIRST_NAMES = [
    "Ava", "Liam", "Mia", "Noah", "Sofia", "Ethan", "Isabella", "Lucas", "Amara",
    "Mateo", "Chloe", "Kai", "Priya", "Diego", "Hannah", "Omar", "Grace", "Leo",
    "Zara", "Jonah", "Elena", "Ravi", "Nora", "Felix", "Maya", "Hugo", "Lina",
    "Theo", "Aisha", "Sam",
]
LAST_NAMES = [
    "Nguyen", "Garcia", "Smith", "Patel", "Kim", "Johnson", "Lopez", "Chen",
    "Brown", "Davis", "Martinez", "Wilson", "Anderson", "Thomas", "Moore",
    "Singh", "Clark", "Lewis", "Walker", "Young",
]
# (city, state)
CITIES = [
    ("Los Angeles", "CA"), ("San Diego", "CA"), ("San Jose", "CA"),
    ("Seattle", "WA"), ("Portland", "OR"), ("Phoenix", "AZ"),
    ("Denver", "CO"), ("Austin", "TX"), ("Dallas", "TX"), ("Chicago", "IL"),
    ("New York", "NY"), ("Boston", "MA"), ("Miami", "FL"), ("Atlanta", "GA"),
]
SEGMENTS = [("Consumer", 0.7), ("Small Business", 0.2), ("Corporate", 0.1)]
CHANNELS = [("Web", 0.55), ("Mobile App", 0.35), ("Marketplace", 0.10)]
RETURN_REASONS = [
    ("Wrong size", 0.30), ("Defective", 0.20), ("Not as described", 0.18),
    ("Changed mind", 0.17), ("Arrived late", 0.08), ("Damaged in shipping", 0.07),
]

SCHEMA_SQL = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    price        REAL NOT NULL,   -- current list price in USD
    cost         REAL NOT NULL    -- unit cost in USD
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL,
    city         TEXT NOT NULL,
    state        TEXT NOT NULL,   -- two-letter US state code
    segment      TEXT NOT NULL,   -- Consumer / Small Business / Corporate
    signup_date  TEXT NOT NULL    -- ISO date YYYY-MM-DD
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   TEXT NOT NULL,   -- ISO date YYYY-MM-DD
    status       TEXT NOT NULL,   -- Delivered / Shipped / Processing / Cancelled
    channel      TEXT NOT NULL    -- Web / Mobile App / Marketplace
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL   -- price actually paid per unit (after discounts)
);

CREATE TABLE returns (
    return_id     INTEGER PRIMARY KEY,
    order_item_id INTEGER NOT NULL REFERENCES order_items(order_item_id),
    return_date   TEXT NOT NULL,  -- ISO date YYYY-MM-DD
    reason        TEXT NOT NULL,
    refund_amount REAL NOT NULL   -- USD refunded
);

CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_returns_item ON returns(order_item_id);
"""


def _weighted(rng: random.Random, pairs):
    values, weights = zip(*pairs)
    return rng.choices(values, weights=weights, k=1)[0]


def build_database(path: str | Path = DEFAULT_DB_PATH, *, today: date | None = None,
                   seed: int = 42, n_customers: int = 400, n_orders: int = 3000) -> Path:
    """Create (or overwrite) the sample database at *path* and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = random.Random(seed)
    today = today or date.today()
    start = date(today.year - 1, 1, 1)          # ~ 22 months of history
    span_days = (today - start).days

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)

    # Products
    for pid, (name, cat, price, _) in enumerate(PRODUCTS, start=1):
        cost = round(price * rng.uniform(0.35, 0.6), 2)
        conn.execute("INSERT INTO products VALUES (?,?,?,?,?)", (pid, name, cat, price, cost))

    # Customers
    for cid in range(1, n_customers + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        city, state = rng.choice(CITIES)
        signup = start + timedelta(days=rng.randint(0, max(span_days - 30, 1)))
        email = f"{first.lower()}.{last.lower()}{cid}@example.com"
        conn.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
            (cid, f"{first} {last}", email, city, state, _weighted(rng, SEGMENTS), signup.isoformat()),
        )

    # Orders, items, returns. Sales grow over time and peak in Nov/Dec.
    item_id = 0
    return_id = 0
    for oid in range(1, n_orders + 1):
        while True:
            d = start + timedelta(days=rng.randint(0, span_days))
            growth = 0.55 + 0.45 * ((d - start).days / span_days)
            seasonal = 1.6 if d.month in (11, 12) else (1.2 if d.month in (6, 7) else 1.0)
            if rng.random() < growth * seasonal / 1.6:
                break
        age_days = (today - d).days
        if age_days < 3:
            status = _weighted(rng, [("Processing", 0.7), ("Shipped", 0.3)])
        elif age_days < 8:
            status = _weighted(rng, [("Shipped", 0.6), ("Delivered", 0.35), ("Cancelled", 0.05)])
        else:
            status = _weighted(rng, [("Delivered", 0.95), ("Cancelled", 0.05)])

        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?)",
            (oid, rng.randint(1, n_customers), d.isoformat(), status, _weighted(rng, CHANNELS)),
        )

        for pid in rng.sample(range(1, len(PRODUCTS) + 1), k=rng.choice([1, 1, 1, 2, 2, 3])):
            item_id += 1
            _, _, price, return_p = PRODUCTS[pid - 1]
            qty = rng.choice([1, 1, 1, 1, 2, 2, 3])
            discount = rng.choice([1.0, 1.0, 1.0, 0.9, 0.85]) if d.month != 11 else rng.choice([0.8, 0.75, 0.9])
            unit_price = round(price * discount, 2)
            conn.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (item_id, oid, pid, qty, unit_price))

            # Returns only for delivered orders old enough to be returned
            if status == "Delivered" and age_days >= 5:
                # Return rates drift a little over time so month-to-month answers differ
                drift = 1.0 + 0.35 * ((pid * 7 + d.month * 3) % 5 - 2) / 2
                if rng.random() < return_p * drift:
                    return_id += 1
                    ret_date = d + timedelta(days=rng.randint(4, min(30, age_days)))
                    if ret_date > today:
                        ret_date = today
                    reason = _weighted(rng, RETURN_REASONS)
                    if pid <= 4 or pid in (10, 11, 13):          # apparel & footwear: sizing issues
                        reason = rng.choice([reason, "Wrong size"])
                    refund = round(unit_price * qty * rng.choice([1.0, 1.0, 1.0, 0.5]), 2)
                    conn.execute(
                        "INSERT INTO returns VALUES (?,?,?,?,?)",
                        (return_id, item_id, ret_date.isoformat(), reason, refund),
                    )

    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    p = build_database()
    print(f"Wrote {p}")
