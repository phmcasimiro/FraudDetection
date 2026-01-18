# db.py
# Script para interação com o banco de dados SQLite
# Elaborado por: phmcasimiro

import sqlite3
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.getcwd(), "data", "fraud_detection.db")


def get_connection():
    """Establishes a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def save_data(df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
    """
    Saves a DataFrame to a SQLite table.

    Args:
        df: The DataFrame to save.
        table_name: The name of the table.
        if_exists: How to behave if the table already exists ('fail', 'replace', 'append').
    """
    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        logger.info(f"Data saved to table '{table_name}' in {DB_PATH}")
    except Exception as e:
        logger.error(f"Error saving data to table '{table_name}': {e}")
        raise e
    finally:
        conn.close()


def load_data(table_name: str) -> pd.DataFrame:
    """
    Loads data from a SQLite table into a DataFrame.

    Args:
        table_name: The name of the table.

    Returns:
        pd.DataFrame: The loaded data.
    """
    conn = get_connection()
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        logger.info(f"Data loaded from table '{table_name}' in {DB_PATH}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from table '{table_name}': {e}")
        raise e
    finally:
        conn.close()
