import uvicorn
from app.main import create_app
from app.core.config import settings
import logging
import asyncio
from app.db.database import init_db, cleanup_db
from app.core.logging_config import setup_logging
import signal
import sys
from typing import Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI

# إعداد التسجيل
setup_logging()
logger = logging.getLogger(__name__)

def main() -> None:
    """النقطة الرئيسية لتشغيل التطبيق"""
    try:
        # تكوين خيارات Uvicorn
        uvicorn_config = {
            "factory": True,
            "app": "app.main:create_app",
            "host": "0.0.0.0",
            "port": 8000,
            "reload": settings.app.debug,
            "workers": 1,
            "loop": "auto",
            "log_config": None,
            "access_log": True,
            "proxy_headers": True,
            "forwarded_allow_ips": "*",
            "timeout_keep_alive": 120,
            "limit_concurrency": 1000,
            "backlog": 2048,
            "lifespan": "on"
        }
        
        # تشغيل الخادم
        logger.info(f"بدء تشغيل الخادم على http://0.0.0.0:8000")
        uvicorn.run(**uvicorn_config)
        
    except Exception as e:
        logger.error(f"خطأ في تشغيل التطبيق: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()