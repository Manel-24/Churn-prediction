from sqlalchemy import create_engine, MetaData
from databases import Database
import os
from dotenv import load_dotenv
import sys

# Add the environment module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environment'))
from environment_config import env  # This will set the DATABASE_URL

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# MySQL-specific database configuration
database = Database(DATABASE_URL)
metadata = MetaData()

# Sync version for creating tables with MySQL-specific settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=False,          # Set to True for SQL debugging
    connect_args={
        "charset": "utf8mb4",
        "autocommit": True
    }
)
