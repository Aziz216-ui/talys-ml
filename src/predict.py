"""
Generation des predictions futures a partir du modele entraine.
"""

import pandas as pd
from datetime import timedelta


def predict_future(model, daily_clean, features, horizon_days=30):
    """Prevision recursive sur horizon_days jours.

    Chaque prediction est reinjectee dans l historique pour calculer les
    lags et moyennes mobiles du jour suivant (sinon ces features resteraient
    figees sur les dernieres valeurs connues pour tout l horizon).
    """
    historique = daily_clean["montant_total"].tolist()
    derniere_date = daily_clean["date_transaction"].max()

    resultats = []
    for i in range(1, horizon_days + 1):
        date = derniere_date + timedelta(days=i)

        ligne = {
            "jour_semaine": date.dayofweek,
            "jour_mois": date.day,
            "mois": date.month,
            "annee": date.year,
            "jour_annee": date.dayofyear,
            "lag_1": historique[-1],
            "lag_7": historique[-7],
            "lag_30": historique[-30],
            "moyenne_mobile_7": sum(historique[-7:]) / 7,
            "moyenne_mobile_30": sum(historique[-30:]) / 30,
        }

        X_futur = pd.DataFrame([ligne])[features]
        prediction = model.predict(X_futur)[0]

        historique.append(prediction)
        resultats.append({"date_transaction": date, "montant_predit": prediction})

    return pd.DataFrame(resultats)
