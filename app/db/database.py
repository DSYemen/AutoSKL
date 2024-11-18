from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy import event, text
from app.core.config import settings
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import asyncio
from functools import wraps
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Prometheus متريكس
DB_CONNECTIONS = Gauge('db_active_connections', 'Number of active database connections')
DB_POOL_SIZE = Gauge('db_pool_size', 'Database connection pool size')
DB_OPERATIONS = Counter('db_operations_total', 'Total database operations', ['operation'])
DB_ERRORS = Counter('db_errors_total', 'Total database errors', ['error_type'])

# تكوين المحرك حسب نوع قاعدة البيانات
if settings.database.url.startswith('sqlite'):
    database_url = settings.database.url.replace('sqlite:', 'sqlite+aiosqlite:')
    engine = create_async_engine(
        database_url,
        echo=settings.database.echo,
        poolclass=AsyncAdaptedQueuePool,
        pool_pre_ping=True,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
        connect_args={
            'timeout': settings.database.idle_timeout / 1000,  # تحويل إلى ثواني
            'check_same_thread': False
        }
    )
else:
    engine = create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_pre_ping=True,
        pool_recycle=settings.database.pool_recycle,
        connect_args={
            'statement_timeout': settings.database.statement_timeout,
            'idle_in_transaction_session_timeout': settings.database.idle_timeout
        }
    )

# إنشاء صانع الجلسات
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# القاعدة الأساسية للنماذج
Base = declarative_base()

class DatabaseManager:
    """مدير قاعدة البيانات"""
    def __init__(self) -> None:
        self.engine = engine
        self.connection_retries = settings.database.connection_retries
        self.retry_interval = settings.database.retry_interval
        
    async def check_connection(self) -> bool:
        """التحقق من اتصال قاعدة البيانات مع إعادة المحاولة"""
        for attempt in range(self.connection_retries):
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    return True
            except Exception as e:
                logger.warning(f"فشل محاولة الاتصال {attempt + 1}: {str(e)}")
                if attempt < self.connection_retries - 1:
                    await asyncio.sleep(self.retry_interval)
                else:
                    DB_ERRORS.labels(error_type='connection').inc()
                    logger.error("��شل الاتصال بقاعدة البيانات بعد كل المحاولات")
        return False
        
    async def get_active_connections_count(self) -> int:
        """الحصول على عدد الاتصالات النشطة"""
        try:
            pool = engine.pool
            count = pool.size() if pool else 0
            DB_CONNECTIONS.set(count)
            return count
        except Exception as e:
            logger.error(f"خطأ في الحصول على عدد الاتصالات: {str(e)}")
            return 0
            
    async def get_table_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الجداول"""
        try:
            async with AsyncSessionLocal() as session:
                result = {}
                for table in Base.metadata.tables:
                    count = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    result[table] = count.scalar()
                return result
        except Exception as e:
            logger.error(f"خطأ في الحصول على إحصائيات الجداول: {str(e)}")
            DB_ERRORS.labels(error_type='stats').inc()
            return {}

db_manager = DatabaseManager()

@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """الحصول على جلسة قاعدة البيانات"""
    session = AsyncSessionLocal()
    try:
        DB_OPERATIONS.labels(operation='session_start').inc()
        yield session
        await session.commit()
        DB_OPERATIONS.labels(operation='session_commit').inc()
    except Exception as e:
        await session.rollback()
        DB_OPERATIONS.labels(operation='session_rollback').inc()
        DB_ERRORS.labels(error_type='session').inc()
        logger.error(f"خطأ في جلسة قاعدة البيانات: {str(e)}")
        raise
    finally:
        await session.close()
        DB_OPERATIONS.labels(operation='session_close').inc()

async def init_db() -> None:
    """تهيئة قاعدة البيانات"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            DB_OPERATIONS.labels(operation='init').inc()
            logger.info("تم إنشاء جداول قاعدة البيانات بنجاح")
    except Exception as e:
        DB_ERRORS.labels(error_type='init').inc()
        logger.error(f"خطأ في تهيئة قاعدة البيانات: {str(e)}")
        raise

async def cleanup_db() -> None:
    """تنظيف موارد قاعدة البيانات"""
    try:
        await engine.dispose()
        DB_OPERATIONS.labels(operation='cleanup').inc()
        logger.info("تم تنظيف موارد قاعدة البيانات")
    except Exception as e:
        DB_ERRORS.labels(error_type='cleanup').inc()
        logger.error(f"خطأ في تنظيف موارد قاعدة البيانات: {str(e)}")
        raise

def db_session_decorator():
    """مزخرف لإدارة جلسة قاعدة البيانات"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with get_db() as session:
                return await func(*args, session=session, **kwargs)
        return wrapper
    return decorator
