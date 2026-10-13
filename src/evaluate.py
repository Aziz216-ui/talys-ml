"""
Evaluation du modele et comparaison avec la version en production.
"""

import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


from datetime import datetime


def compute_metrics(y_true, y_pred):
    """Calcule MAE, RMSE, MAPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {"date_run": now_str, "mae": mae, "rmse": rmse, "mape": mape}



def is_better(new_metrics, old_metrics_path, metric="mape"):
    """Compare le nouveau modele a l ancien via l historique sauvegarde."""
    try:
        with open(old_metrics_path) as f:
            history = json.load(f)
        old_value = history[-1][metric]
    except (FileNotFoundError, IndexError, KeyError):
        return True

    return new_metrics[metric] < old_value


def save_metrics(metrics, metrics_path):
    """Ajoute les metriques du run courant a l historique JSON."""
    try:
        with open(metrics_path) as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append({"run_id": f"Run {len(history) + 1}", **metrics})

    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
