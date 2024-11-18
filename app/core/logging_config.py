import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable, TypeVar, ParamSpec
import traceback
from functools import wraps
from app.core.config import settings

P = ParamSpec('P')
T = TypeVar('T')

class JsonFormatter(logging.Formatter):
    """منسق مخصص للسجلات بتنسيق JSON"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread
        }
        
        # إضافة معلومات الاستثناء إذا وجدت
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # إضافة البيانات الإضافية
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
            
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging() -> None:
    """إعداد نظام التسجيل"""
    # إنشاء مجلد السجلات
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # تكوين المسجل الرئيسي
    root_logger = logging.getLogger()
    
    # تحديد مستوى التسجيل
    default_level = "INFO"
    log_level = getattr(settings.logging, 'level', default_level)
    if isinstance(log_level, str):
        log_level = log_level.upper()
    root_logger.setLevel(log_level)
    
    # إعداد المنسق
    if getattr(settings.logging, 'json_format', True):
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            getattr(settings.logging, 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    # معالج الملف الدوار حسب الحجم
    size_handler = RotatingFileHandler(
        getattr(settings.logging, 'file', 'logs/app.log'),
        maxBytes=getattr(settings.logging, 'max_size', 10485760),
        backupCount=getattr(settings.logging, 'backup_count', 5),
        encoding='utf-8'
    )
    size_handler.setFormatter(formatter)
    root_logger.addHandler(size_handler)
    
    # معالج الملف الدوار حسب الوقت (يومياً)
    time_handler = TimedRotatingFileHandler(
        str(log_dir / 'app_daily.log'),
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    time_handler.setFormatter(formatter)
    root_logger.addHandler(time_handler)
    
    # معالج وحدة التحكم
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # إعداد مستويات التسجيل للمكونات المختلفة
    logging.getLogger('sqlalchemy.engine').setLevel(
        logging.INFO if getattr(settings.logging, 'log_sql', False) else logging.WARNING
    )
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # تسجيل بدء التطبيق
    root_logger.info(
        "تم بدء تشغيل التطبيق",
        extra={'extra_data': {'version': settings.app.version}}
    )

def get_logger(name: str) -> logging.Logger:
    """الحصول على مسجل مكون"""
    return logging.getLogger(f"app.{name}")

def with_logging_context(context: Dict[str, Any]) -> Callable:
    """مزخرف لإضافة سياق للتسجيل"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            logger = get_logger(func.__module__)
            try:
                result = await func(*args, **kwargs)
                logger.info(
                    f"تم تنفيذ {func.__name__} بنجاح",
                    extra={'extra_data': context}
                )
                return result
            except Exception as e:
                logger.error(
                    f"خطأ في تنفيذ {func.__name__}: {str(e)}",
                    extra={'extra_data': {**context, 'error': str(e)}},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator 