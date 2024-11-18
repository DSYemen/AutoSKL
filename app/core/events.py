from typing import Callable, Dict, Any, Optional
from fastapi import FastAPI
import logging
from app.core.logging_config import setup_logging
from app.db.database import init_db, cleanup_db, db_manager
from app.ml.model_updater import start_model_update_scheduler, stop_model_update_scheduler
from app.utils.cache import cache_manager
from app.utils.alerts import alert_manager
import psutil
import asyncio
from datetime import datetime, timedelta
import prometheus_client
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Prometheus متريكس
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
DISK_USAGE = Gauge('disk_usage_bytes', 'Disk usage in bytes')
MODELS_COUNT = Gauge('models_count', 'Number of loaded models')
DB_CONNECTIONS = Gauge('db_connections', 'Number of database connections')
UPTIME = Gauge('app_uptime_seconds', 'Application uptime in seconds')

class SystemMonitor:
    """مراقب موارد النظام"""
    @staticmethod
    async def collect_metrics() -> None:
        """جمع مقاييس النظام"""
        try:
            # استخدام الذاكرة
            memory = psutil.Process().memory_info()
            MEMORY_USAGE.set(memory.rss)
            
            # استخدام المعالج
            CPU_USAGE.set(psutil.cpu_percent())
            
            # استخدام القرص
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.used)
            
            try:
                # عدد النماذج المحملة
                stats = await db_manager.get_table_stats()
                models_count = stats.get('models', 0)
                if isinstance(models_count, (dict, list)):
                    models_count = len(models_count)
                MODELS_COUNT.set(models_count)
                
                # عدد اتصالات قاعدة البيانات
                db_connections = await db_manager.get_active_connections_count()
                DB_CONNECTIONS.set(db_connections)
            except Exception as db_error:
                logger.warning(f"فشل جمع مقاييس قاعدة البيانات: {str(db_error)}")
            
            # وقت التشغيل
            if hasattr(app_start_time, 'timestamp'):
                uptime = (datetime.utcnow() - app_start_time).total_seconds()
                UPTIME.set(uptime)
            
        except Exception as e:
            logger.error(f"خطأ في جمع مقاييس النظام: {str(e)}")

def create_start_app_handler(app: FastAPI) -> Callable:
    """إنشاء معالج بدء التطبيق"""
    async def start_app() -> None:
        try:
            # تخزين وقت بدء التشغيل
            global app_start_time
            app_start_time = datetime.utcnow()
            app.state.start_time = app_start_time
            
            # إعداد التسجيل
            setup_logging()
            logger.info("تم إعداد التسجيل")
            
            # تهيئة قاعدة البيانات
            await init_db()
            logger.info("تم تهيئة قاعدة البيانات")
            
            # التحقق من اتصال قاعدة البيانات
            if await db_manager.check_connection():
                logger.info("تم الاتصال بقاعدة البيانات بنجاح")
            else:
                logger.error("فشل الاتصال بقاعدة البيانات")
                
            # بدء جدولة تحديث النماذج
            await start_model_update_scheduler()
            logger.info("تم ��دء جدولة تحديث النماذج")
            
            # اختبار اتصال Redis
            if await cache_manager.test_connection():
                logger.info("تم الاتصال بـ Redis بنجاح")
            else:
                logger.warning("Redis غير متوفر، سيتم استخدام التخزين المؤقت المحلي")
                
            # بدء مراقبة النظام
            app.state.monitor_task = asyncio.create_task(
                periodic_system_monitoring(interval=60)  # كل دقيقة
            )
            
            # إرسال تنبيه بدء التشغيل
            try:
                await alert_manager.send_startup_alert()
            except Exception as e:
                logger.warning(f"فشل إرسال تنبيه بدء التشغيل: {str(e)}")
            
        except Exception as e:
            logger.error(f"خطأ أثناء بدء التطبيق: {str(e)}")
            raise
            
    return start_app

def create_stop_app_handler(app: FastAPI) -> Callable:
    """إنشاء معالج إيقاف التطبيق"""
    async def stop_app() -> None:
        try:
            # إيقاف مراقبة النظام
            if hasattr(app.state, 'monitor_task'):
                app.state.monitor_task.cancel()
                try:
                    await app.state.monitor_task
                except asyncio.CancelledError:
                    pass
                    
            # إيقاف جدولة تحديث النماذج
            await stop_model_update_scheduler()
            logger.info("تم إيقاف جدولة تحديث النماذج")
            
            # مسح الذاكرة المؤقتة
            await cache_manager.clear()
            logger.info("تم مسح الذاكرة المؤقتة")
            
            # تنظيف موارد قاعدة البيانات
            await cleanup_db()
            logger.info("تم تنظيف موارد قاعدة البيانات")
            
            # إغلاق اتصال Redis
            await cache_manager.close()
            logger.info("تم إغلاق اتصال Redis")
            
            # إرسال تنبيه إيقاف التشغيل
            await alert_manager.send_shutdown_alert()
            
        except Exception as e:
            logger.error(f"خطأ أثناء إيقاف التطبيق: {str(e)}")
            raise
            
    return stop_app

async def periodic_system_monitoring(interval: int = 60) -> None:
    """مراقبة دورية لموارد النظام"""
    while True:
        try:
            await SystemMonitor.collect_metrics()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"خطأ في مراقبة النظام: {str(e)}")
            await asyncio.sleep(interval)

# متغير عام لتخزين وقت بدء التشغيل
app_start_time: datetime = datetime.utcnow()