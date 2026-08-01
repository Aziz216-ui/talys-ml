"""
Construction des features temporelles, lags et moyennes mobiles
pour le modèle de forecasting.
"""

FEATURES = ["jour_semaine", "jour_mois", "mois", "annee", "jour_annee",
            "lag_1", "lag_7", "lag_30", "moyenne_mobile_7", "moyenne_mobile_30"]

TARGET = "montant_total"


def add_features(daily):
    """Ajoute les features temporelles, lags et moyennes mobiles a daily."""
    daily = daily.copy()

    daily["jour_semaine"] = daily["date_transaction"].dt.dayofweek
    daily["jour_mois"] = daily["date_transaction"].dt.day
    daily["mois"] = daily["date_transaction"].dt.month
    daily["annee"] = daily["date_transaction"].dt.year
    daily["jour_annee"] = daily["date_transaction"].dt.dayofyear

    daily["lag_1"] = daily["montant_total"].shift(1)
    daily["lag_7"] = daily["montant_total"].shift(7)
    daily["lag_30"] = daily["montant_total"].shift(30)

    daily["moyenne_mobile_7"] = daily["montant_total"].rolling(window=7).mean()
    daily["moyenne_mobile_30"] = daily["montant_total"].rolling(window=30).mean()

    return daily.dropna().reset_index(drop=True)
