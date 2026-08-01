"""
Script principal : orchestre le pipeline complet de reentrainement.
Charge les donnees, entraine, evalue, compare a l ancien modele,
sauvegarde si meilleur, et genere les predictions futures.
"""

import joblib
import os

from data_loader import load_raw_data, build_daily_aggregation
from features import add_features, FEATURES
from train import split_train_test, train_model
from evaluate import compute_metrics, is_better, save_metrics
from predict import generate_future_features, predict_future

DATA_DIR = "data/raw"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model_current.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_history.json")


def main():
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

    print("4. Entrainement + tuning...")
    model, best_params = train_model(X_train, y_train)
    print(f"   Meilleurs parametres : {best_params}")

    print("5. Evaluation...")
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(f"   MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%")

    print("6. Comparaison avec l ancien modele...")
    if is_better(metrics, METRICS_PATH):
        joblib.dump(model, MODEL_PATH)
        save_metrics(metrics, METRICS_PATH)
        print("   Nouveau modele deploye.")
    else:
        print("   Ancien modele conserve (nouveau moins bon).")
        model = joblib.load(MODEL_PATH)

    print("7. Generation des predictions futures...")
    futur = generate_future_features(daily_clean, horizon_days=30)
    predictions = predict_future(model, futur, FEATURES)
    predictions.to_csv(os.path.join(DATA_DIR, "predictions_ventes.csv"), index=False)
    print("   Predictions sauvegardees dans data/raw/predictions_ventes.csv")


if __name__ == "__main__":
    main()
