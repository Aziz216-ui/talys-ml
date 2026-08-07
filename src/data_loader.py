"""
Chargement des données depuis les exports CSV du data mart MySQL
et agrégation journalière des ventes.
"""

import pandas as pd


def load_raw_data(fact_sales_path, dim_date_path, dim_product_path):
    
    fact_sales = pd.read_csv(fact_sales_path)
    dim_date = pd.read_csv(dim_date_path)
    dim_product = pd.read_csv(dim_product_path)
    return fact_sales, dim_date, dim_product


def build_daily_aggregation(fact_sales, dim_date, dim_product):
    
    df = fact_sales.merge(dim_date, on='ID_Temps').merge(dim_product, on='ID_Product')

    daily = df.groupby('date_transaction').agg(
        nombre_ventes=('ID_Fact_Sales', 'count'),
        montant_total=('Price', 'sum')
    ).reset_index()

    daily['date_transaction'] = pd.to_datetime(daily['date_transaction'])
    daily = daily.sort_values('date_transaction').reset_index(drop=True)

    return daily
