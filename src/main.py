"""
Pipeline de reentrainement, decoupe en etapes reutilisables :
chargement (MySQL ou CSV) -> entrainement -> comparaison au modele en production
sur la meme periode de test -> deploiement si meilleur -> predictions -> publication.

Les etapes sont orchestrees par Prefect (flows/retrain_flow.py) ;
`python src/main.py` les enchaine directement, sans orchestrateur.
"""

import joblib
import os
from datetime import datetime
import pandas as pd
import mlflow
import mlflow.xgboost


from data_loader import load_raw_data, build_daily_aggregation, load_daily_from_mysql
from db import use_mysql, get_engine
from features import add_features, FEATURES
from train import split_train_test, train_model
from evaluate import compute_metrics, save_metrics
from predict import predict_future

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_current.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_history.json")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")


def load_daily():
    if use_mysql():
        print("   Source : Data Mart MySQL (v_ventes_quotidiennes)")
        return load_daily_from_mysql(get_engine())

    print("   Source : fichiers CSV (data/raw)")
    fact_sales, dim_date, dim_product = load_raw_data(
        os.path.join(DATA_DIR, "fact_sales.csv"),
        os.path.join(DATA_DIR, "dim_date.csv"),
        os.path.join(DATA_DIR, "dim_product.csv"),
    )
    return build_daily_aggregation(fact_sales, dim_date, dim_product)


def train_and_select(daily_clean, trigger_source):
    """Entraine un challenger, le compare au champion sur la meme periode de test et deploie le meilleur."""
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
    mlflow.set_experiment("Talys_Sales_Forecasting")
    X_train, X_test, y_train, y_test = split_train_test(daily_clean)

    with mlflow.start_run():
        mlflow.set_tag("trigger", trigger_source)
        model, best_params = train_model(X_train, y_train)
        print(f"   Meilleurs parametres : {best_params}")

        metrics = compute_metrics(y_test, model.predict(X_test))
        print(f"   Challenger : MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%")

        mlflow.log_params(best_params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("date_run", str(metrics["date_run"]))
        try:
            mlflow.xgboost.log_model(model, name="xgb_model")
        except Exception:
            mlflow.xgboost.log_model(model, artifact_path="xgb_model")

        champion_metrics = None
        if os.path.exists(MODEL_PATH):
            champion = joblib.load(MODEL_PATH)
            champion_metrics = compute_metrics(y_test, champion.predict(X_test))
            mlflow.log_metric("champion_mape", champion_metrics["mape"])
            print(f"   Champion   : MAE={champion_metrics['mae']:.2f}  RMSE={champion_metrics['rmse']:.2f}  MAPE={champion_metrics['mape']:.2f}%")

        promoted = champion_metrics is None or metrics["mape"] < champion_metrics["mape"]
        mlflow.set_tag("promu", str(promoted))
        if promoted:
            joblib.dump(model, MODEL_PATH)
            save_metrics(metrics, METRICS_PATH)
            print("   Nouveau modele deploye.")
        else:
            print("   Ancien modele conserve (nouveau moins bon).")
            model = champion

    production = metrics if promoted else champion_metrics
    return model, {
        "date_run": datetime.now(),
        "trigger_source": trigger_source,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "champion_mape": champion_metrics["mape"] if champion_metrics else None,
        "production_mae": production["mae"],
        "production_rmse": production["rmse"],
        "production_mape": production["mape"],
        "promu": promoted,
    }


def build_outputs(model, daily_clean):
    """Predictions a 30 jours + serie combinee reel / predit (aussi exportees en CSV)."""
    predictions = predict_future(model, daily_clean, FEATURES, horizon_days=30)
    predictions["montant_predit"] = predictions["montant_predit"].astype(float)
    predictions.to_csv(os.path.join(DATA_DIR, "predictions_ventes.csv"), index=False)

    historique = daily_clean[["date_transaction", "montant_total"]].rename(columns={"montant_total": "Montant"})
    historique["Type"] = "Reel"
    futur = predictions.rename(columns={"montant_predit": "Montant"})
    futur["Type"] = "Predit"
    combine = pd.concat([historique, futur], ignore_index=True)
    combine.to_csv(os.path.join(DATA_DIR, "ventes_combinees.csv"), index=False)
    return predictions, combine


def publish_to_mysql(predictions, combine, run_record):
    """Publie predictions, serie combinee et suivi du run dans MySQL (sources de Power BI)."""
    # Une seule transaction : Power BI ne voit jamais un etat partiellement publie
    with get_engine().begin() as conn:
        conn.exec_driver_sql("DELETE FROM ml_predictions")
        conn.exec_driver_sql("DELETE FROM ml_ventes_combinees")
        predictions.assign(generated_at=datetime.now()).to_sql("ml_predictions", conn, if_exists="append", index=False)
        combine.to_sql("ml_ventes_combinees", conn, if_exists="append", index=False, chunksize=5000)
        pd.DataFrame([run_record]).to_sql("ml_training_runs", conn, if_exists="append", index=False)


def generate_report():
    """Redige le rapport executif LLM et le publie dans MySQL si la source est MySQL."""
    from reports.report_generator import generate_executive_report
    content = generate_executive_report()
    if use_mysql():
        pd.DataFrame([{"generated_at": datetime.now(), "markdown_content": content}]).to_sql(
            "ml_reports", get_engine(), if_exists="append", index=False
        )
    return content


def main(trigger_source="manuel"):
    print("1. Chargement des donnees...")
    daily = load_daily()
    print(f"   {len(daily)} jours, derniere date {daily['date_transaction'].max().date()}")

    print("2. Feature engineering...")
    daily_clean = add_features(daily)

    print("3. Entrainement et comparaison au modele en production...")
    model, run_record = train_and_select(daily_clean, trigger_source)
    run_record["derniere_date_donnees"] = daily["date_transaction"].max().date()
    run_record["nb_ventes"] = int(daily["nombre_ventes"].sum())

    print("4. Predictions a 30 jours...")
    predictions, combine = build_outputs(model, daily_clean)
    if use_mysql():
        publish_to_mysql(predictions, combine, run_record)
        print("   Predictions et suivi du run publies dans MySQL.")

    print("5. Rapport decisionnel LLM...")
    try:
        generate_report()
    except Exception as e:
        print(f"   Note sur le rapport LLM: {e}")

    return run_record


if __name__ == "__main__":
    main()
