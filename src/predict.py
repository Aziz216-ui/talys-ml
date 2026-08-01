"""
Generation des predictions futures a partir du modele entraine.
"""

import pandas as pd
from datetime import timedelta


def generate_future_features(daily_clean, horizon_days=30):
    """Construit les features pour les jours a venir (lags figes sur les dernieres valeurs connues)."""
    derniere_date = daily_clean["date_transaction"].max()
    dates_futures = pd.date_range(start=derniere_date + timedelta(days=1), periods=horizon_days)

    futur = pd.DataFrame({"date_transaction": dates_futures})
    futur["jour_semaine"] = futur["date_transaction"].dt.dayofweek
    futur["jour_mois"] = futur["date_transaction"].dt.day
    futur["mois"] = futur["date_transaction"].dt.month
    futur["annee"] = futur["date_transaction"].dt.year
    futur["jour_annee"] = futur["date_transaction"].dt.dayofyear

    futur["lag_1"] = daily_clean["montant_total"].iloc[-1]
    futur["lag_7"] = daily_clean["montant_total"].iloc[-7]
    futur["lag_30"] = daily_clean["montant_total"].iloc[-30]
    futur["moyenne_mobile_7"] = daily_clean["montant_total"].iloc[-7:].mean()
    futur["moyenne_mobile_30"] = daily_clean["montant_total"].iloc[-30:].mean()

    return futur


def predict_future(model, futur, features):
    """Applique le modele sur les features futures."""
    futur = futur.copy()
    futur["montant_predit"] = model.predict(futur[features])
    return futur[["date_transaction", "montant_predit"]]
