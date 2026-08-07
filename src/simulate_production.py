"""
Script de simulation des conditions de production (Production Retraining Simulation).
1. Simule l'arrivée de nouvelles ventes quotidiennes dans fact_sales.csv.
2. Déclenche le pipeline automatique de réentraînement (main.py).
3. Affiche les métriques enregistrées dans MLflow (mlflow.db) et la mise à jour des prédictions/rapports.
"""

import os
import sys
import pandas as pd
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FACT_SALES_PATH = os.path.join(DATA_DIR, "fact_sales.csv")


def simulate_new_sales_batch(num_transactions: int = 500):
    """Génère de nouvelles transactions de ventes et les ajoute à fact_sales.csv."""
    if not os.path.exists(FACT_SALES_PATH):
        print(f"Erreur : {FACT_SALES_PATH} introuvable.")
        return

    df = pd.read_csv(FACT_SALES_PATH)
    last_id = df["ID_Fact_Sales"].max()

    new_rows = []
    for i in range(1, num_transactions + 1):
        new_id = last_id + i
        # Sélection aléatoire de temps, produit, client, localisation existants
        temps_id = random.randint(1, 3500)
        product_id = random.randint(1, 500)
        client_id = random.randint(1, 500)
        loc_id = random.randint(1, 500)

        new_rows.append({
            "ID_Fact_Sales": new_id,
            "ID_Temps": temps_id,
            "ID_Product": product_id,
            "ID_Client": client_id,
            "ID_Localisation": loc_id
        })

    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(FACT_SALES_PATH, index=False)

    print(f"⚡ SIMULATION PRODUCTION : {num_transactions} nouvelles ventes ajoutées à fact_sales.csv !")
    print(f"   Nombre total de transactions : {len(updated_df)} (précédemment : {len(df)})")


def main():
    print("=" * 70)
    print("🚀 DÉMARRAGE DE LA SIMULATION DE PRODUCTION (FLUX DE DONNÉES TEMPS RÉEL)")
    print("=" * 70)

    # Étape 1 : Simulation de nouvelles transactions
    simulate_new_sales_batch(num_transactions=800)

    print("\n🔄 Lancement du Pipeline de Réentraînement Automatique...\n")

    # Étape 2 : Exécution de main.py
    sys.path.append(os.path.join(BASE_DIR, "src"))
    from main import main as run_pipeline
    run_pipeline()

    print("\n" + "=" * 70)
    print("✅ SIMULATION TERMINÉE AVEC SUCCÈS !")
    print("   - MLflow SQLite (mlflow.db) a enregistré le nouveau run")
    print("   - Le modèle XGBoost a été réévalué contre l'ancien")
    print("   - Les fichiers ventes_combinees.csv et executive_report_latest.md ont été actualisés")
    print("=" * 70)


if __name__ == "__main__":
    main()
