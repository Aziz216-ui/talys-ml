"""
Connexion au Data Mart MySQL (talys_datamart), local (XAMPP) ou heberge (ex. Aiven).
La source de donnees du pipeline est choisie via DATA_SOURCE (mysql | csv).

Variables d'environnement (.env en local, Secrets sur Streamlit Cloud) :
  MYSQL_URL           mysql+pymysql://user:motdepasse@hote:port/talys_datamart
  MYSQL_SSL_CA        chemin du certificat CA (connexion chiffree)
  MYSQL_SSL_CA_PEM    ou contenu du certificat CA (pratique pour les Secrets Streamlit)
"""

import os
import tempfile
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/talys_datamart")


def use_mysql():
    return os.getenv("DATA_SOURCE", "csv").lower() == "mysql"


def ssl_ca_path():
    """Chemin du certificat CA, ecrit dans un fichier temporaire s'il est fourni en texte."""
    pem = os.getenv("MYSQL_SSL_CA_PEM")
    if pem:
        path = os.path.join(tempfile.gettempdir(), "talys_mysql_ca.pem")
        with open(path, "w") as f:
            f.write(pem)
        return path
    path = os.getenv("MYSQL_SSL_CA")
    if path and not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return path


@lru_cache(maxsize=None)
def get_engine(url=None):
    ca = ssl_ca_path()
    connect_args = {"ssl": {"ca": ca}} if ca else {}
    return create_engine(url or MYSQL_URL, pool_pre_ping=True, pool_recycle=280, connect_args=connect_args)
