import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
from config import Config
import logging
import time

logger = logging.getLogger(__name__)

class Database:
    """Database connection and query utilities with connection pooling"""
    
    def __init__(self):
        self.pool = None
        self._init_pool()
        
    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = PooledDB(
                creator=pymysql,
                maxconnections=10,  # Maximum connections in pool
                mincached=2,        # Minimum idle connections
                maxcached=5,        # Maximum idle connections
                maxshared=3,        # Maximum shared connections
                blocking=True,      # Block if no connections available
                maxusage=None,      # Reuse connections indefinitely
                setsession=[],
                ping=1,             # Check connection before using (1 = default)
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                cursorclass=DictCursor,
                charset='utf8mb4',
                autocommit=False
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            return self.pool.connection()
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise
    
    def execute_query(self, query, params=None, fetch=True):
        """Execute a query and return results"""
        connection = None
        try:
            start_time = time.time()
            connection = self.get_connection()
            
            with connection.cursor() as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    result = cursor.fetchall()
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:  # Log slow queries
                        logger.warning(f"Slow query ({elapsed:.2f}s): {query[:100]}")
                    return result
                else:
                    connection.commit()
                    return cursor.lastrowid
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            if connection:
                connection.close()  # Return to pool
    
    def execute_many(self, query, params_list):
        """Execute multiple queries with different parameters"""
        connection = None
        try:
            connection = self.get_connection()
            
            with connection.cursor() as cursor:
                cursor.executemany(query, params_list)
                connection.commit()
                return cursor.rowcount
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Batch execution failed: {e}")
            raise
        finally:
            if connection:
                connection.close()  # Return to pool

# Global database instance
db = Database()

def get_db():
    """Get database instance"""
    return db

def init_db():
    """Initialize database with schema"""
    try:
        connection = pymysql.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        
        with connection.cursor() as cursor:
            # Read and execute schema file
            with open('../database/schema.sql', 'r', encoding='utf-8') as f:
                schema = f.read()
                # Split by semicolon and execute each statement
                statements = [s.strip() for s in schema.split(';') if s.strip()]
                for statement in statements:
                    cursor.execute(statement)
        
        connection.commit()
        connection.close()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
