import asyncio
import logging
from slipstream.analytics.storage_layer import AnalyticsStorage, DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnectionTest")

async def test_connection():
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="slipstream_analytics",
        username="postgres",
        password="postgres"
    )
    
    storage = AnalyticsStorage(config)
    
    logger.info("Connecting to database...")
    try:
        await storage.connect()
        logger.info("✅ Connection successful!")
        
        logger.info("Creating tables...")
        await storage.create_tables()
        logger.info("✅ Tables created/verified!")
        
        await storage.disconnect()
        logger.info("Disconnected.")
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if not success:
        exit(1)
