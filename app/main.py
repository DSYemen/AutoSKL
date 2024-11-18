from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Callable, Awaitable, TypeVar, Union
from collections.abc import Sequence
import logging
import time
import prometheus_client
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from typing_extensions import TypeAlias

from app.core.config import settings
from app.core.events import create_start_app_handler, create_stop_app_handler
from app.api.routes import router
from app.db.database import db_manager
from app.utils.exceptions import MLFrameworkError, format_error_response

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
T = TypeVar('T')
HealthStatus: TypeAlias = Dict[str, Any]
SystemStats: TypeAlias = Dict[str, Any]

class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """وسيط لقياس أداء الطلبات"""
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.REQUEST_LATENCY = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'path']
        )
        self.REQUEST_COUNT = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'path', 'status']
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            if response:
                duration = time.time() - start_time
                self.REQUEST_LATENCY.labels(
                    method=request.method,
                    path=request.url.path
                ).observe(duration)
                self.REQUEST_COUNT.labels(
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code
                ).inc()

def create_app() -> FastAPI:
    """إنشاء تطبيق FastAPI"""
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        docs_url=settings.app.docs_url,
        redoc_url=settings.app.redoc_url,
        openapi_tags=[
            {"name": "ML Models", "description": "عمليات النماذج"},
            {"name": "Monitoring", "description": "مراقبة النظام"},
            {"name": "Reports", "description": "إدارة التقارير"}
        ]
    )
    
    # إضافة معالجات الأحداث
    app.add_event_handler("startup", create_start_app_handler(app))
    app.add_event_handler("shutdown", create_stop_app_handler(app))
    
    # إعداد الوسطاء
    app.add_middleware(RequestMetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600
    )
    
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=6
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.app.allowed_hosts
    )
    
    # إنشاء المجلدات المطلوبة
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    for subdir in ['css', 'js', 'images', 'fonts']:
        (static_dir / subdir).mkdir(exist_ok=True)
    
    # تثبيت الملفات الثابتة
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # إعداد القوالب
    templates = Jinja2Templates(directory="app/templates")
    app.state.templates = templates
    
    # إضافة فلاتر Jinja2 مخصصة
    templates.env.filters["datetime"] = format_datetime
    templates.env.filters["number"] = format_number
    templates.env.filters["duration"] = format_duration
    
    @app.get("/", response_model=Dict[str, Any])
    async def index(request: Request) -> Any:
        """الصفحة الرئيسية"""
        try:
            system_stats = await get_system_stats(app)
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "stats": system_stats,
                    "app_name": settings.app.name,
                    "version": settings.app.version
                }
            )
        except Exception as e:
            logger.error(f"خطأ في الصفحة الرئيسية: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل الصفحة الرئيسية"
                },
                status_code=500
            )
    
    @app.get("/health", response_model=HealthStatus)
    async def health_check() -> HealthStatus:
        """فحص صحة النظام"""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'database': await db_manager.check_connection(),
                'version': settings.app.version,
                'uptime': get_uptime(app)
            }
            return health_status
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
    
    # إضافة مسارات API
    app.include_router(router, prefix="/api/v1")
    
    # معالج الأخطاء العام
    @app.exception_handler(MLFrameworkError)
    async def ml_framework_exception_handler(request: Request, exc: MLFrameworkError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=format_error_response(exc)
        )
    
    @app.get("/models/new", response_model=Dict[str, Any])
    async def new_model_page(request: Request) -> Any:
        """صفحة تدريب نموذج جديد"""
        try:
            # تحضير خيارات المعالجة المسبقة من الإعدادات
            preprocessing_options = {
                "missing_values": settings.ml.preprocessing['missing_values'],
                "feature_selection": settings.ml.preprocessing['feature_selection'],
                "scaling": settings.ml.preprocessing['scaling'],
                "encoding": settings.ml.preprocessing['encoding']
            } if hasattr(settings.ml, 'preprocessing') else {
                "missing_values": {"strategy": "iterative"},
                "feature_selection": {"method": "mutual_info"},
                "scaling": {"method": "robust"},
                "encoding": {"method": "onehot"}
            }

            return templates.TemplateResponse(
                "new_model.html",
                {
                    "request": request,
                    "app_name": settings.app.name,
                    "version": settings.app.version,
                    "task_types": settings.ml.task_types,
                    "preprocessing_options": preprocessing_options,
                    "training_options": settings.ml.training
                }
            )
        except Exception as e:
            logger.error(f"خطأ في صفحة النموذج الجديد: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل صفحة النموذج الجديد"
                },
                status_code=500
            )

    @app.get("/models/{model_id}", response_model=Dict[str, Any])
    async def model_details_page(model_id: str, request: Request) -> Any:
        """صفحة تفاصيل النموذج"""
        try:
            model_info = await db_manager.get_model_info(model_id)
            if not model_info:
                return templates.TemplateResponse(
                    "error.html",
                    {
                        "request": request,
                        "error_message": f"النموذج {model_id} غير موجود"
                    },
                    status_code=404
                )
            
            return templates.TemplateResponse(
                "model_details.html",
                {
                    "request": request,
                    "app_name": settings.app.name,
                    "version": settings.app.version,
                    "model": model_info
                }
            )
        except Exception as e:
            logger.error(f"خطأ في صفحة تفاصيل النموذج: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل صفحة تفاصيل النموذج"
                },
                status_code=500
            )

    @app.get("/models", response_model=Dict[str, Any])
    async def models_page(request: Request) -> Any:
        """صفحة قائمة النماذج"""
        try:
            models = await db_manager.get_all_models()
            return templates.TemplateResponse(
                "models.html",
                {
                    "request": request,
                    "app_name": settings.app.name,
                    "version": settings.app.version,
                    "models": models
                }
            )
        except Exception as e:
            logger.error(f"خطأ في صفحة قائمة النماذج: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل صفحة قائمة النماذج"
                },
                status_code=500
            )

    @app.get("/monitoring", response_model=Dict[str, Any])
    async def monitoring_page(request: Request) -> Any:
        """صفحة المراقبة"""
        try:
            return templates.TemplateResponse(
                "monitoring.html",
                {
                    "request": request,
                    "app_name": settings.app.name,
                    "version": settings.app.version,
                    "monitoring_config": settings.ml.monitoring
                }
            )
        except Exception as e:
            logger.error(f"خطأ في صفحة المراقبة: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل صفحة المراقبة"
                },
                status_code=500
            )

    @app.get("/reports", response_model=Dict[str, Any])
    async def reports_page(request: Request) -> Any:
        """صفحة التقارير"""
        try:
            return templates.TemplateResponse(
                "reports.html",
                {
                    "request": request,
                    "app_name": settings.app.name,
                    "version": settings.app.version,
                    "reporting_config": settings.reporting
                }
            )
        except Exception as e:
            logger.error(f"خطأ في صفحة التقارير: {str(e)}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "حدث خطأ في تحميل صفحة التقارير"
                },
                status_code=500
            )
    
    return app

async def get_system_stats(app: FastAPI) -> SystemStats:
    """جمع إحصائيات النظام"""
    stats = {
        'database_connected': False,
        'models_count': 0,
        'predictions_count': 0,
        'uptime': get_uptime(app)
    }
    
    try:
        stats['database_connected'] = await db_manager.check_connection()
        if stats['database_connected']:
            db_stats = await db_manager.get_table_stats()
            stats.update({
                'models_count': len(db_stats.get('models', [])),
                'predictions_count': len(db_stats.get('predictions', []))
            })
    except Exception as e:
        logger.warning(f"فشل جلب إحصائيات النظام: {str(e)}")
        
    return stats

def get_uptime(app: FastAPI) -> float:
    """حساب وقت تشغيل التطبيق"""
    if hasattr(app.state, 'start_time'):
        return (datetime.utcnow() - app.state.start_time).total_seconds()
    return 0.0

def format_datetime(value: Union[str, datetime]) -> str:
    """تنسيق التاريخ والوقت"""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime("%Y-%m-%d %H:%M:%S")

def format_number(value: Union[int, float, str]) -> str:
    """تنسيق الأرقام"""
    try:
        return "{:,}".format(float(value))
    except (ValueError, TypeError):
        return str(value)

def format_duration(seconds: Union[int, float]) -> str:
    """تنسيق المدة الزمنية"""
    try:
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        
        parts = []
        if days > 0:
            parts.append(f"{days} يوم")
        if hours > 0:
            parts.append(f"{hours} ساعة")
        if minutes > 0:
            parts.append(f"{minutes} دقيقة")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} ثانية")
            
        return " و ".join(parts)
    except (ValueError, TypeError):
        return str(seconds)

# إنشاء نسخة من التطبيق
app = create_app()