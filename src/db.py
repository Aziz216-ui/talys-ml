"""
Connexion au Data Mart MySQL (talys_datamart).
La source de donnees du pipeline est choisie via DATA_SOURCE (mysql | csv).
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/talys_datamart")


def use_mysql():
    return os.getenv("DATA_SOURCE", "csv").lower() == "mysql"


def get_engine():
    return create_engine(MYSQL_URL, pool_pre_ping=True)
