"""
Orchestration Prefect du reentrainement Talys-Trade.

Le flow verifie d'abord si de nouvelles journees de ventes sont arrivees dans le
Data Mart MySQL depuis le dernier entrainement ; sinon il s'arrete sans rien faire.

Usage :
  python flows/retrain_flow.py                     # une execution maintenant
  python flows/retrain_flow.py --force             # reentraine meme sans nouvelles donnees
  python flows/retrain_flow.py --serve             # deploiement planifie (tous les jours a 02:00)
  python flows/retrain_flow.py --serve --interval 2   # demo : verification toutes les 2 minutes

Declenchement evenementiel (ex. a la fin du job Talend) :
  prefect deployment run "talys-retrain/talys-retrain-planifie"
"""

import argparse
import json
import os
import sys
from datetime import datetime

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
os.environ["DATA_SOURCE"] = "mysql"

import main as pipeline  # noqa: E402
from features import add_features  # noqa: E402
from db import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

STATE_PATH = os.path.join(BASE_DIR, "models", "training_state.json")


def read_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@task(name="Detecter les nouvelles donnees", retries=2, retry_delay_seconds=30)
def detect_new_data():
    state = read_state()
    with get_engine().connect() as conn:
        last_id, last_date = conn.execute(text(
            "SELECT MAX(f.ID_Fact_Sales), MAX(d.date_transaction) FROM fact_sales f JOIN dim_date d USING (ID_Temps)"
        )).one()
        new_days = conn.execute(text(
            "SELECT COUNT(DISTINCT d.date_transaction) FROM fact_sales f JOIN dim_date d USING (ID_Temps) "
            "WHERE f.ID_Fact_Sales > :last_id"
        ), {"last_id": state.get("last_fact_id", 0)}).scalar()
    print(f"Derniere vente #{last_id} ({last_date}) ; dernier entrainement sur #{state.get('last_fact_id', 0)} "
          f"-> {new_days} nouvelle(s) journee(s)")
    return {"last_fact_id": last_id, "last_date": str(last_date), "new_days": new_days}


@task(name="Charger les ventes (MySQL)", retries=2, retry_delay_seconds=30)
def load_data():
    daily = pipeline.load_daily()
    return daily, add_features(daily)


@task(name="Entrainer et comparer au champion")
def train(daily_clean, trigger_source):
    return pipeline.train_and_select(daily_clean, trigger_source)


@task(name="Generer les predictions a 30 jours")
def forecast(model, daily_clean):
    return pipeline.build_outputs(model, daily_clean)


@task(name="Publier dans MySQL", retries=2, retry_delay_seconds=30)
def publish(predictions, combine, run_record):
    pipeline.publish_to_mysql(predictions, combine, run_record)


@task(name="Rediger le rapport LLM", retries=1, retry_delay_seconds=60)
def report():
    pipeline.generate_report()


@flow(name="talys-retrain", log_prints=True)
def retrain_flow(min_new_days: int = 1, force: bool = False, trigger_source: str = "prefect"):
    # Verrou MySQL : jamais deux reentrainements en parallele (planification + declenchement Talend)
    with get_engine().connect() as lock_conn:
        if not lock_conn.execute(text("SELECT GET_LOCK('talys_retrain', 0)")).scalar():
            print("Un reentrainement est deja en cours : execution ignoree.")
            return {"reentraine": False}
        try:
            return run_retrain(min_new_days, force, trigger_source)
        finally:
            lock_conn.execute(text("SELECT RELEASE_LOCK('talys_retrain')"))


def run_retrain(min_new_days, force, trigger_source):
    data = detect_new_data()
    if not force and data["new_days"] < min_new_days:
        print(f"Moins de {min_new_days} nouvelle(s) journee(s) : pas de reentrainement.")
        return {"reentraine": False, **data}

    daily, daily_clean = load_data()
    model, run_record = train(daily_clean, trigger_source)
    run_record["derniere_date_donnees"] = daily["date_transaction"].max().date()
    run_record["nb_ventes"] = int(daily["nombre_ventes"].sum())

    predictions, combine = forecast(model, daily_clean)
    publish(predictions, combine, run_record)
    try:
        report()
    except Exception as e:
        print(f"Rapport LLM non genere : {e}")

    with open(STATE_PATH, "w") as f:
        json.dump({"last_fact_id": data["last_fact_id"], "last_date": data["last_date"],
                   "last_training": datetime.now().isoformat(timespec="seconds"),
                   "promu": run_record["promu"], "production_mape": run_record["production_mape"]}, f, indent=2)

    verdict = "Nouveau modele deploye" if run_record["promu"] else "Modele en production conserve"
    create_markdown_artifact(
        key="resultat-reentrainement",
        description="Resultat du dernier reentrainement",
        markdown=(
            f"### {verdict}\n\n"
            f"| | MAPE |\n|---|---|\n"
            f"| Challenger | {run_record['mape']:.2f} % |\n"
            f"| Champion | {run_record['champion_mape'] or float('nan'):.2f} % |\n"
            f"| **En production** | **{run_record['production_mape']:.2f} %** |\n\n"
            f"Donnees jusqu'au {run_record['derniere_date_donnees']} ({data['new_days']} nouvelle(s) journee(s)) ; "
            f"predictions du {predictions['date_transaction'].min():%Y-%m-%d} au {predictions['date_transaction'].max():%Y-%m-%d}."
        ),
    )
    print(verdict)
    return {"reentraine": True, "promu": run_record["promu"], "production_mape": run_record["production_mape"], **data}


def purge_stale_runs():
    """Annule les executions planifiees restees en attente pendant que le deploiement etait arrete,
    pour ne pas les rejouer toutes d'un coup au redemarrage."""
    import asyncio
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import (
        FlowFilter, FlowFilterName, FlowRunFilter, FlowRunFilterState, FlowRunFilterStateType,
    )
    from prefect.client.schemas.objects import StateType
    from prefect.states import Cancelled

    async def purge():
        async with get_client() as client:
            total = 0
            while True:
                runs = await client.read_flow_runs(
                    flow_filter=FlowFilter(name=FlowFilterName(any_=["talys-retrain"])),
                    flow_run_filter=FlowRunFilter(state=FlowRunFilterState(type=FlowRunFilterStateType(
                        any_=[StateType.SCHEDULED, StateType.PENDING, StateType.RUNNING]))),
                    limit=200,
                )
                if not runs:
                    break
                for run in runs:
                    await client.set_flow_run_state(run.id, Cancelled(message="Purge au redemarrage du deploiement"), force=True)
                total += len(runs)
            if total:
                print(f"{total} execution(s) en attente annulee(s).")

    asyncio.run(purge())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Prefect de reentrainement Talys-Trade.")
    parser.add_argument("--serve", action="store_true", help="demarre le deploiement planifie")
    parser.add_argument("--interval", type=float, help="minutes entre deux executions (au lieu du cron quotidien)")
    parser.add_argument("--cron", default="0 2 * * *", help="planification cron (defaut : tous les jours a 02:00)")
    parser.add_argument("--min-new-days", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.serve:
        purge_stale_runs()
        schedule = {"interval": args.interval * 60} if args.interval else {"cron": args.cron}
        retrain_flow.serve(
            name="talys-retrain-planifie",
            parameters={"min_new_days": args.min_new_days},
            tags=["talys", "mlops"],
            limit=1,
            **schedule,
        )
    else:
        retrain_flow(min_new_days=args.min_new_days, force=args.force, trigger_source="manuel")
