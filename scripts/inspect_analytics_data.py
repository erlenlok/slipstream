import asyncio
import asyncpg
import os
from datetime import datetime

DB_CONFIG = {
    "user": "postgres",
    "password": "password",  # Default from setup script
    "database": "slipstream_analytics",
    "host": "localhost",
    "port": 5432,
}

# Override with env if needed, but sticking to defaults known from previous setup
# "password" was used in setup_analytics_db.sh

async def main():
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        print("Connected to database.")
    except Exception as e:
        print(f"Connection failed: {e}")
        # Try 'postgres' password just in case (config.py had 'postgres')
        try:
            DB_CONFIG["password"] = "postgres"
            conn = await asyncpg.connect(**DB_CONFIG)
            print("Connected to database (password: postgres).")
        except Exception as e2:
            print(f"Connection failed again: {e2}")
            return

    # List Tables
    print("\n--- Existing Tables ---")
    tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    for t in tables:
        print(f"- {t['table_name']}")

    # 1. Check Trade Events
    print("\n--- Trade Events Table ---")
    try:
        row = await conn.fetchrow("SELECT count(*) FROM trade_events")
        count = row[0]
        print(f"Total Trade Events: {count}")

        if count > 0:
            rows = await conn.fetch("SELECT * FROM trade_events ORDER BY timestamp DESC LIMIT 5")
            for r in rows:
                print(f"[{r['timestamp']}] {r['symbol']} {r['side']} {r['quantity']} @ {r['price']} ({r['trade_type']})")
    except asyncpg.exceptions.UndefinedTableError:
        print("Table 'trade_events' does not exist.")

    # 2. Check Snapshots
    print("\n--- Performance Snapshots Table ---")
    try:
        row = await conn.fetchrow("SELECT count(*) FROM performance_snapshots")
        print(f"Total Snapshots: {row[0]}")
    except asyncpg.exceptions.UndefinedTableError:
        print("Table 'performance_snapshots' does not exist.")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
