# migrate_to_db.py
# Script para migrar dados do CSV para o banco de dados SQLite
# Elaborado por: phmcasimiro

import pandas as pd
import os
import logging
from src.data.db import save_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_data():
    csv_path = "src/data/creditcard.csv"

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        return

    logger.info(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows.")

        logger.info("Saving to database...")
        save_data(df, "transactions", if_exists="replace")
        logger.info("Migration completed successfully.")

    except Exception as e:
        logger.error(f"Migration failed: {e}")


if __name__ == "__main__":
    migrate_data()
