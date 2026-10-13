"""
Script principal : orchestre le pipeline complet de reentrainement.
Charge les donnees (MySQL ou CSV), entraine, compare au modele en production
sur la meme periode de test, deploie si meilleur, et genere les predictions futures.
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


def write_outputs_to_mysql(predictions, combine, run_record):
    """Publie les predictions et le suivi du run dans MySQL (lus par Power BI)."""
    engine = get_engine()
    now = datetime.now()
    preds = predictions.assign(generated_at=now)
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM ml_predictions")
        conn.exec_driver_sql("DELETE FROM ml_ventes_combinees")
    preds.to_sql("ml_predictions", engine, if_exists="append", index=False)
    combine.to_sql("ml_ventes_combinees", engine, if_exists="append", index=False, chunksize=5000)
    pd.DataFrame([run_record]).to_sql("ml_training_runs", engine, if_exists="append", index=False)


def main(trigger_source="manuel"):
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
    mlflow.set_experiment("Talys_Sales_Forecasting")

    print("1. Chargement des donnees...")
    daily = load_daily()
    print(f"   {len(daily)} jours, derniere date {daily['date_transaction'].max().date()}")

    print("2. Feature engineering...")
    daily_clean = add_features(daily)

    print("3. Split train/test...")
    X_train, X_test, y_train, y_test = split_train_test(daily_clean)

    with mlflow.start_run():
        mlflow.set_tag("trigger", trigger_source)
        print("4. Entrainement + tuning...")
        model, best_params = train_model(X_train, y_train)
        print(f"   Meilleurs parametres : {best_params}")

        print("5. Evaluation...")
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(f"   MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%")

        print("6. Enregistrement MLflow...")
        mlflow.log_params(best_params)
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric_metrics)
        if "date_run" in metrics:
            mlflow.set_tag("date_run", str(metrics["date_run"]))
        try:
            mlflow.xgboost.log_model(model, name="xgb_model")
        except Exception:
            mlflow.xgboost.log_model(model, artifact_path="xgb_model")

        print("7. Comparaison avec le modele en production (meme periode de test)...")
        champion_mape = None
        if os.path.exists(MODEL_PATH):
            champion = joblib.load(MODEL_PATH)
            champion_mape = compute_metrics(y_test, champion.predict(X_test))["mape"]
            mlflow.log_metric("champion_mape", champion_mape)
            print(f"   Champion : MAPE={champion_mape:.2f}%  |  Challenger : MAPE={metrics['mape']:.2f}%")

        promoted = champion_mape is None or metrics["mape"] < champion_mape
        mlflow.set_tag("promu", str(promoted))
        if promoted:
            joblib.dump(model, MODEL_PATH)
            save_metrics(metrics, METRICS_PATH)
            print("   Nouveau modele deploye.")
        else:
            print("   Ancien modele conserve (nouveau moins bon).")
            model = champion

    print("8. Generation des predictions futures...")
    predictions = predict_future(model, daily_clean, FEATURES, horizon_days=30)
    predictions.to_csv(os.path.join(DATA_DIR, "predictions_ventes.csv"), index=False)
    print("   Predictions sauvegardees dans data/raw/predictions_ventes.csv")

    print("9. Creation du fichier combine reel + predit...")
    historique = daily_clean[["date_transaction", "montant_total"]].copy()
    historique["Type"] = "Reel"
    historique = historique.rename(columns={"montant_total": "Montant"})

    futur_export = predictions.copy()
    futur_export["Type"] = "Predit"
    futur_export = futur_export.rename(columns={"montant_predit": "Montant"})

    combine = pd.concat([historique, futur_export], ignore_index=True)
    combine.to_csv(os.path.join(DATA_DIR, "ventes_combinees.csv"), index=False)
    print("   Fichier ventes_combinees.csv genere.")

    run_record = {
        "date_run": datetime.now(),
        "trigger_source": trigger_source,
        "derniere_date_donnees": daily["date_transaction"].max().date(),
        "nb_ventes": int(daily["nombre_ventes"].sum()),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "champion_mape": champion_mape,
        "promu": promoted,
    }
    if use_mysql():
        write_outputs_to_mysql(predictions, combine, run_record)
        print("   Predictions et suivi du run publies dans MySQL.")

    print("10. Generation du rapport decisionnel LLM...")
    try:
        import sys
        sys.path.append(os.path.join(BASE_DIR, "src"))
        from reports.report_generator import generate_executive_report
        generate_executive_report()
    except Exception as e:
        print(f"   Note sur le rapport LLM: {e}")

    return run_record


if __name__ == "__main__":
    main()
