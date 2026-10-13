-- Data Mart Talys-Trade (schema en etoile) + tables de sortie du pipeline ML
CREATE DATABASE IF NOT EXISTS talys_datamart CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE talys_datamart;

-- Dimensions
CREATE TABLE IF NOT EXISTS dim_date (
    ID_Temps INT PRIMARY KEY,
    date_transaction DATE NOT NULL UNIQUE,
    jour INT,
    mois INT,
    trimestre INT,
    annee INT,
    date_transaction_reelle VARCHAR(64)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dim_product (
    ID_Product INT PRIMARY KEY,
    Product_Name VARCHAR(64),
    Ph INT,
    Price INT,
    Branche VARCHAR(8),
    Categorie VARCHAR(32),
    Sous_Categorie VARCHAR(16)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dim_client (
    ID_Client INT PRIMARY KEY,
    FirstName VARCHAR(64),
    LastName VARCHAR(64),
    Age INT,
    Gender VARCHAR(16)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dim_localisation (
    ID_Localisation INT PRIMARY KEY,
    City VARCHAR(64),
    State VARCHAR(64),
    State_ID VARCHAR(8),
    Street VARCHAR(128)
) ENGINE=InnoDB;

-- Table de faits
CREATE TABLE IF NOT EXISTS fact_sales (
    ID_Fact_Sales INT PRIMARY KEY,
    ID_Temps INT NOT NULL,
    ID_Product INT NOT NULL,
    ID_Client INT NOT NULL,
    ID_Localisation INT NOT NULL,
    CONSTRAINT fk_sales_date FOREIGN KEY (ID_Temps) REFERENCES dim_date (ID_Temps),
    CONSTRAINT fk_sales_product FOREIGN KEY (ID_Product) REFERENCES dim_product (ID_Product),
    CONSTRAINT fk_sales_client FOREIGN KEY (ID_Client) REFERENCES dim_client (ID_Client),
    CONSTRAINT fk_sales_localisation FOREIGN KEY (ID_Localisation) REFERENCES dim_localisation (ID_Localisation)
) ENGINE=InnoDB;

-- Serie temporelle journaliere utilisee par le modele
CREATE OR REPLACE VIEW v_ventes_quotidiennes AS
SELECT d.date_transaction,
       COUNT(f.ID_Fact_Sales) AS nombre_ventes,
       SUM(p.Price) AS montant_total
FROM fact_sales f
JOIN dim_date d ON d.ID_Temps = f.ID_Temps
JOIN dim_product p ON p.ID_Product = f.ID_Product
GROUP BY d.date_transaction;

-- Sorties du pipeline ML (lues par Power BI)
CREATE TABLE IF NOT EXISTS ml_predictions (
    date_transaction DATE PRIMARY KEY,
    montant_predit DOUBLE NOT NULL,
    generated_at DATETIME NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ml_ventes_combinees (
    date_transaction DATE NOT NULL,
    Montant DOUBLE NOT NULL,
    Type VARCHAR(8) NOT NULL,
    PRIMARY KEY (date_transaction, Type)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ml_training_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date_run DATETIME NOT NULL,
    trigger_source VARCHAR(32),
    derniere_date_donnees DATE,
    nb_ventes INT,
    mae DOUBLE,
    rmse DOUBLE,
    mape DOUBLE,
    champion_mape DOUBLE,
    production_mae DOUBLE,
    production_rmse DOUBLE,
    production_mape DOUBLE,
    promu BOOLEAN NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ml_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    generated_at DATETIME NOT NULL,
    markdown_content MEDIUMTEXT NOT NULL
) ENGINE=InnoDB;

-- Historique des runs pour la page MLOps de Power BI
CREATE OR REPLACE VIEW v_metrics_history AS
SELECT CONCAT('Run ', id) AS run_id,
       date_run,
       trigger_source,
       derniere_date_donnees,
       mae, rmse, mape,
       production_mae, production_rmse, production_mape,
       promu
FROM ml_training_runs;

-- Dernier rapport executif LLM (page IA Copilot de Power BI)
CREATE OR REPLACE VIEW v_latest_report AS
SELECT 'executive_report_latest.md' AS file, generated_at, markdown_content
FROM ml_reports
ORDER BY id DESC
LIMIT 1;
