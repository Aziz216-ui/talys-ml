"""
Initialise le Data Mart MySQL talys_datamart (base cible = MYSQL_URL).

Usage :
  python src/load_mysql.py                          # depuis les exports CSV (data/raw)
  python src/load_mysql.py --from-url mysql+pymysql://root:@127.0.0.1:3306/talys_datamart
                                                    # copie une base MySQL existante
                                                    # (Data Mart + sorties ML), ex. local -> cloud
"""

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine, text

from db import BASE_DIR, MYSQL_URL, get_engine

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
SCHEMA_PATH = os.path.join(BASE_DIR, "sql", "schema.sql")

# Ordre de chargement : dimensions avant la table de faits (cles etrangeres)
TABLES = ["dim_date", "dim_product", "dim_client", "dim_localisation", "fact_sales"]
ML_TABLES = ["ml_predictions", "ml_ventes_combinees", "ml_training_runs", "ml_reports"]


def run_schema(server_url):
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        statements = [s.strip() for s in f.read().split(";") if s.strip()]
    with get_engine(server_url).begin() as conn:
        for stmt in statements:
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            if lines:
                conn.execute(text("\n".join(lines)))


def read_source(table, source_engine):
    if source_engine is not None:
        return pd.read_sql(f"SELECT * FROM {table}", source_engine)
    df = pd.read_csv(os.path.join(DATA_DIR, f"{table}.csv"))
    if table == "dim_date":
        df["date_transaction"] = pd.to_datetime(df["date_transaction"]).dt.date
    return df


def main():
    parser = argparse.ArgumentParser(description="Initialise le Data Mart MySQL talys_datamart.")
    parser.add_argument("--from-url", help="URL SQLAlchemy d'une base MySQL source a copier")
    args = parser.parse_args()

    server_url = MYSQL_URL.rsplit("/", 1)[0] + "/"
    print("1. Creation du schema...")
    run_schema(server_url)

    source_engine = create_engine(args.from_url) if args.from_url else None
    tables = TABLES + (ML_TABLES if source_engine is not None else [])

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in reversed(tables):
            conn.execute(text(f"DELETE FROM {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    print(f"2. Chargement des tables depuis {'MySQL ' + args.from_url.split('@')[-1] if source_engine is not None else 'les CSV'}...")
    for table in tables:
        df = read_source(table, source_engine)
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=2000, method="multi")
        print(f"   {table:<20} {len(df):>7} lignes")

    with engine.connect() as conn:
        n, last = conn.execute(text(
            "SELECT COUNT(*), MAX(date_transaction) FROM fact_sales f JOIN dim_date d USING (ID_Temps)"
        )).one()
    print(f"3. Data Mart pret : {n} ventes, derniere date {last}")


if __name__ == "__main__":
    main()
