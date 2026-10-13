"""
Watcher de reentrainement automatique.

Surveille le Data Mart MySQL et relance le pipeline des que de nouvelles ventes
sont arrivees depuis le dernier entrainement (filigrane = MAX(ID_Fact_Sales)).

Usage :
  python src/watcher.py                 # surveille en continu (toutes les 60 s)
  python src/watcher.py --interval 30   # verification toutes les 30 s
  python src/watcher.py --once          # une seule verification (pour un planificateur)
"""

import argparse
import json
import os
import time
from datetime import datetime

from sqlalchemy import text

os.environ["DATA_SOURCE"] = "mysql"

from db import BASE_DIR, get_engine
from main import main as run_pipeline

STATE_PATH = os.path.join(BASE_DIR, "models", "training_state.json")


def read_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_fact_id": 0}


def write_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def current_watermark(engine):
    with engine.connect() as conn:
        last_id, last_date = conn.execute(text(
            "SELECT MAX(f.ID_Fact_Sales), MAX(d.date_transaction) "
            "FROM fact_sales f JOIN dim_date d USING (ID_Temps)"
        )).one()
    return last_id or 0, last_date


def check_once(engine):
    state = read_state()
    last_id, last_date = current_watermark(engine)
    now = datetime.now().strftime("%H:%M:%S")

    if last_id <= state["last_fact_id"]:
        print(f"[{now}] Aucune nouvelle vente (derniere : #{last_id}, {last_date}).")
        return False

    new_rows = last_id - state["last_fact_id"]
    print(f"[{now}] {new_rows} nouvelle(s) vente(s) detectee(s) jusqu'au {last_date} -> reentrainement")
    record = run_pipeline(trigger_source="watcher")
    write_state({
        "last_fact_id": last_id,
        "last_date": last_date,
        "last_training": datetime.now().isoformat(timespec="seconds"),
        "promu": record["promu"],
        "mape": record["mape"],
    })
    verdict = "nouveau modele deploye" if record["promu"] else "modele en production conserve"
    print(f"[{datetime.now():%H:%M:%S}] Reentrainement termine : {verdict}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reentraine automatiquement le modele a l'arrivee de nouvelles ventes.")
    parser.add_argument("--interval", type=float, default=60, help="secondes entre deux verifications")
    parser.add_argument("--once", action="store_true", help="une seule verification puis arret")
    args = parser.parse_args()

    engine = get_engine()
    if args.once:
        check_once(engine)
        return

    print(f"Watcher demarre : verification de talys_datamart toutes les {args.interval:.0f} s (Ctrl+C pour arreter)")
    while True:
        try:
            check_once(engine)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] Erreur : {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
