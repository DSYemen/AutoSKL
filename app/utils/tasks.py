from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, ParamSpec
from celery import Celery, Task
from celery.schedules import crontab
from celery.signals import task_failure, task_success, task_retry
from app.core.config import settings
from app.ml.model_updater import model_updater
from app.ml.monitoring import model_monitor
from app.reporting.report_generator import report_generator
from app.utils.alerts import alert_manager
from app.db.database import get_db
import logging
from datetime import datetime
import traceback
from functools import wraps
from prometheus_client import Counter, Histogram
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
P = ParamSpec('P')
T = TypeVar('T')
TaskResult: TypeAlias = Dict[str, Any]

# Prometheus متريكس
TASK_DURATION = Histogram('task_duration_seconds', 'Task duration in seconds', ['task_name'])
TASK_COUNT = Counter('task_count', 'Task count', ['task_name', 'status'])

# إنشاء تطبيق Celery
celery_app = Celery(
    'ml_framework',
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend
)

# تكوين Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # حد زمني للمهمة (1 ساعة)
    worker_prefetch_multiplier=1,  # تحسين استهلاك الذاكرة
    worker_max_tasks_per_child=1000,  # إعادة تشغيل العامل بعد 1000 مهمة
)

class BaseTask(Task):
    """فئة أساسية للمهام مع تتبع وتسجيل محسن"""
    
    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """معالجة فشل المهمة"""
        TASK_COUNT.labels(task_name=self.name, status='failed').inc()
        
        error_details = {
            'task_id': task_id,
            'args': args,
            'kwargs': kwargs,
            'exception': str(exc),
            'traceback': traceback.format_exc()
        }
        
        logger.error(
            f"فشل المهمة {self.name}",
            extra={'error_details': error_details}
        )
        
        # إرسال تنبيه
        alert_manager.send_error_alert(
            'task_failure',
            str(exc),
            None,
            settings.alerts.recipients
        )
        
    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """معالجة نجاح المهمة"""
        TASK_COUNT.labels(task_name=self.name, status='success').inc()
        logger.info(f"اكتملت المهمة {self.name} بنجاح", extra={'task_id': task_id})
        
    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """معالجة إعادة محاولة المهمة"""
        TASK_COUNT.labels(task_name=self.name, status='retry').inc()
        logger.warning(
            f"إعادة محاولة المهمة {self.name}",
            extra={
                'task_id': task_id,
                'exception': str(exc)
            }
        )

def task_with_metrics(name: Optional[str] = None) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """مزخرف لإضافة قياسات للمهام"""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                TASK_DURATION.labels(task_name=name or func.__name__).observe(duration)
                return result
            except Exception as e:
                TASK_COUNT.labels(task_name=name or func.__name__, status='error').inc()
                raise e
        return wrapper
    return decorator

@celery_app.task(base=BaseTask, bind=True, max_retries=3)
@task_with_metrics(name="model_retrain")
async def model_retrain_task(self,
                           model_id: str,
                           new_data: Dict[str, Any]) -> TaskResult:
    """مهمة إعادة تدريب النموذج"""
    try:
        logger.info(f"بدء مهمة إعادة تدريب النموذج {model_id}")
        async with get_db() as db:
            result = await model_updater.update_model_if_needed(
                model_id,
                new_data,
                db
            )
            
        logger.info(f"اكتمال مهمة إعادة تدريب النموذج {model_id}")
        return {
            'status': 'success',
            'model_id': model_id,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"خطأ في مهمة إعادة تدريب النموذج: {str(e)}")
        await alert_manager.send_error_alert(
            'model_retrain_task_error',
            str(e),
            model_id,
            settings.alerts.recipients
        )
        raise self.retry(exc=e, countdown=300)

@celery_app.task(base=BaseTask, bind=True)
@task_with_metrics(name="generate_report")
async def generate_report_task(self,
                             model_id: str,
                             report_type: str) -> str:
    """مهمة توليد التقرير"""
    try:
        logger.info(f"بدء مهمة توليد تقرير {report_type} للنموذج {model_id}")
        
        if report_type == 'model':
            report_path = await report_generator.generate_model_report(
                model_id=model_id,
                model_info=await model_manager.get_model_info(model_id),
                evaluation_results=await model_evaluator.get_latest_evaluation(model_id),
                feature_importance=await model_evaluator.get_feature_importance(model_id)
            )
        elif report_type == 'monitoring':
            report_path = await report_generator.generate_monitoring_report(
                model_id=model_id,
                monitoring_data=await model_monitor.get_monitoring_summary(model_id)
            )
        else:
            raise ValueError(f"نوع التقرير غير معروف: {report_type}")
            
        logger.info(f"اكتمال مهمة توليد التقرير: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"خطأ في مهمة توليد التقرير: {str(e)}")
        await alert_manager.send_error_alert(
            'report_generation_error',
            str(e),
            model_id,
            settings.alerts.recipients
        )
        raise

# جدولة المهام الدورية
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender: Celery, **kwargs: Any) -> None:
    # تحديث النماذج يومياً
    sender.add_periodic_task(
        crontab(hour=2, minute=0),  # 2 AM
        model_retrain_task.s(),
        name='daily_model_update'
    )
    
    # توليد تقارير المراقبة أسبوعياً
    sender.add_periodic_task(
        crontab(day_of_week=1, hour=3, minute=0),  # 3 AM الاثنين
        generate_report_task.s(report_type='monitoring'),
        name='weekly_monitoring_report'
    )