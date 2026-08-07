"""
Script principal : orchestre le pipeline complet de reentrainement.
Charge les donnees, entraine, evalue, compare a l ancien modele,
sauvegarde si meilleur, et genere les predictions futures.
"""

import joblib
import os
import pandas as pd
import mlflow
import mlflow.xgboost


from data_loader import load_raw_data, build_daily_aggregation
from features import add_features, FEATURES
from train import split_train_test, train_model
from evaluate import compute_metrics, is_better, save_metrics
from predict import predict_future

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_current.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_history.json")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")


def main():
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
    mlflow.set_experiment("Talys_Sales_Forecasting")

    print("1. Chargement des donnees...")
    fact_sales, dim_date, dim_product = load_raw_data(
        os.path.join(DATA_DIR, "fact_sales.csv"),
        os.path.join(DATA_DIR, "dim_date.csv"),
        os.path.join(DATA_DIR, "dim_product.csv"),
    )
    daily = build_daily_aggregation(fact_sales, dim_date, dim_product)

    print("2. Feature engineering...")
    daily_clean = add_features(daily)

    print("3. Split train/test...")
    X_train, X_test, y_train, y_test = split_train_test(daily_clean)

    with mlflow.start_run():
        print("4. Entrainement + tuning...")
        model, best_params = train_model(X_train, y_train)
        print(f"   Meilleurs parametres : {best_params}")

        print("5. Evaluation...")
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(f"   MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%")

        print("6. Enregistrement MLflow...")
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.xgboost.log_model(model, name="xgb_model")
        except Exception:
            mlflow.xgboost.log_model(model, artifact_path="xgb_model")



        print("7. Comparaison avec l ancien modele...")
        if is_better(metrics, METRICS_PATH):
            joblib.dump(model, MODEL_PATH)
            save_metrics(metrics, METRICS_PATH)
            print("   Nouveau modele deploye.")
        else:
            print("   Ancien modele conserve (nouveau moins bon).")
            model = joblib.load(MODEL_PATH)

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


if __name__ == "__main__":
    main()

