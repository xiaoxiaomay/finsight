"""Database initialization script.

Usage:
    python scripts/setup_db.py          # Create all tables
    python scripts/setup_db.py --drop   # Drop and recreate all tables

Note: If using docker-compose, the schema is applied automatically
via the mounted SQL files. This script is for manual re-initialization
or for environments without docker-compose.
"""

import argparse
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """Create a database connection from environment variables."""
    import os

    database_url = os.getenv("DATABASE_URL", "postgresql://finsight:finsight@localhost:5432/finsight")

    return psycopg2.connect(database_url)


def run_sql_file(conn: psycopg2.extensions.connection, filepath: Path) -> None:
    """Execute a SQL file against the database."""
    sql = filepath.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print(f"  Executed: {filepath.name}")


def drop_all_tables(conn: psycopg2.extensions.connection) -> None:
    """Drop all application tables (for development reset)."""
    tables = [
        "research_reports",
        "news_articles",
        "data_quality_log",
        "document_embeddings",
        "portfolio_snapshots",
        "trades",
        "portfolio_positions",
        "factor_signals",
        "macro_indicators",
        "fundamentals",
        "market_data",
    ]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            print(f"  Dropped: {table}")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize FinSight database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables before creating")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent

    print("Connecting to database...")
    conn = get_connection()

    try:
        if args.drop:
            print("\nDropping existing tables...")
            drop_all_tables(conn)

        print("\nCreating extensions...")
        run_sql_file(conn, scripts_dir / "init_extensions.sql")

        print("\nCreating schema...")
        run_sql_file(conn, scripts_dir / "schema.sql")

        # Verify tables
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            tables = [row[0] for row in cur.fetchall()]

        print(f"\nDatabase ready. Tables ({len(tables)}):")
        for t in tables:
            print(f"  - {t}")

    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
