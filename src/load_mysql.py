"""
Initialise le Data Mart MySQL talys_datamart a partir des exports CSV.
Usage : python src/load_mysql.py   (recree les tables du Data Mart)
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

from db import BASE_DIR, MYSQL_URL

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
SCHEMA_PATH = os.path.join(BASE_DIR, "sql", "schema.sql")

# Ordre de chargement : dimensions avant la table de faits (cles etrangeres)
TABLES = ["dim_date", "dim_product", "dim_client", "dim_localisation", "fact_sales"]


def run_schema(server_url):
    engine = create_engine(server_url)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        statements = [s.strip() for s in f.read().split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            if lines:
                conn.execute(text("\n".join(lines)))


def main():
    server_url = MYSQL_URL.rsplit("/", 1)[0] + "/"
    print("1. Creation du schema...")
    run_schema(server_url)

    engine = create_engine(MYSQL_URL)
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in reversed(TABLES):
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    print("2. Chargement des tables...")
    for table in TABLES:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{table}.csv"))
        if table == "dim_date":
            df["date_transaction"] = pd.to_datetime(df["date_transaction"]).dt.date
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=5000, method="multi")
        print(f"   {table:<17} {len(df):>7} lignes")

    with engine.connect() as conn:
        n, last = conn.execute(text(
            "SELECT COUNT(*), MAX(date_transaction) FROM fact_sales f JOIN dim_date d USING (ID_Temps)"
        )).one()
    print(f"3. Data Mart pret : {n} ventes, derniere date {last}")


if __name__ == "__main__":
    main()
