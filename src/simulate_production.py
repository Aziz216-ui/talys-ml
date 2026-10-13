"""
Simulation de production : de nouvelles journees de ventes arrivent dans le Data Mart MySQL.

Chaque journee simulee :
  - ajoute la date suivante dans dim_date ;
  - insere dans fact_sales un nombre de ventes tire de l'historique recent,
    en reprenant le mix produits / clients / localisations de ventes reelles recentes.

Usage :
  python src/simulate_production.py                 # ajoute 1 journee
  python src/simulate_production.py --days 7        # ajoute 7 journees
  python src/simulate_production.py --days 3 --every 60   # 1 journee toutes les 60 s
  python src/simulate_production.py --retrain       # reentraine juste apres l'ajout
"""

import argparse
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine

HISTORY_DAYS = 90


def add_sales_day(engine, rng):
    with engine.begin() as conn:
        id_temps, last_date = conn.execute(text("SELECT MAX(ID_Temps), MAX(date_transaction) FROM dim_date")).one()
        last_id = conn.execute(text("SELECT MAX(ID_Fact_Sales) FROM fact_sales")).scalar()

        recent = pd.read_sql(text(
            "SELECT f.ID_Product, f.ID_Client, f.ID_Localisation, d.date_transaction "
            "FROM fact_sales f JOIN dim_date d USING (ID_Temps) "
            "WHERE d.date_transaction > :since"
        ), conn, params={"since": last_date - timedelta(days=HISTORY_DAYS)})

        daily_counts = recent.groupby("date_transaction").size().to_numpy()
        n_sales = int(rng.choice(daily_counts))

        new_date = last_date + timedelta(days=1)
        new_id_temps = id_temps + 1
        conn.execute(text(
            "INSERT INTO dim_date (ID_Temps, date_transaction, jour, mois, trimestre, annee, date_transaction_reelle) "
            "VALUES (:id, :d, :jour, :mois, :trim, :annee, :reelle)"
        ), {
            "id": new_id_temps, "d": new_date, "jour": new_date.day, "mois": new_date.month,
            "trim": (new_date.month - 1) // 3 + 1, "annee": new_date.year,
            "reelle": new_date.strftime("%a %b %d 00:00:00 GMT+01:00 %Y"),
        })

        sample = recent.sample(n=n_sales, replace=True, random_state=int(rng.integers(1_000_000)))
        rows = [
            {"id": last_id + i + 1, "t": new_id_temps, "p": int(r.ID_Product),
             "c": int(r.ID_Client), "l": int(r.ID_Localisation)}
            for i, r in enumerate(sample.itertuples())
        ]
        conn.execute(text(
            "INSERT INTO fact_sales (ID_Fact_Sales, ID_Temps, ID_Product, ID_Client, ID_Localisation) "
            "VALUES (:id, :t, :p, :c, :l)"
        ), rows)

        montant = conn.execute(text(
            "SELECT montant_total FROM v_ventes_quotidiennes WHERE date_transaction = :d"
        ), {"d": new_date}).scalar()

    print(f"   + {new_date} : {n_sales} ventes, {float(montant):,.0f} TND")
    return new_date


def main():
    parser = argparse.ArgumentParser(description="Simule l'arrivee de nouvelles journees de ventes dans MySQL.")
    parser.add_argument("--days", type=int, default=1, help="nombre de journees a ajouter")
    parser.add_argument("--every", type=float, default=0, help="secondes entre deux journees (0 = d'un coup)")
    parser.add_argument("--retrain", action="store_true", help="lance le reentrainement juste apres")
    args = parser.parse_args()

    engine = get_engine()
    rng = np.random.default_rng()
    print(f"SIMULATION : ajout de {args.days} journee(s) de ventes dans talys_datamart")
    for i in range(args.days):
        add_sales_day(engine, rng)
        if args.every and i < args.days - 1:
            time.sleep(args.every)

    if args.retrain:
        os.environ["DATA_SOURCE"] = "mysql"
        from main import main as run_pipeline
        run_pipeline(trigger_source="simulation")


if __name__ == "__main__":
    main()
