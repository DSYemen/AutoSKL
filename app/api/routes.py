from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Response, BackgroundTasks, Query, status
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
from sse_starlette.sse import EventSourceResponse
from prometheus_client import Counter, Histogram
from sqlalchemy.ext.asyncio import AsyncSession
import io
import aiofiles
import psutil
import os

from app.core.config import settings
from app.ml.data_processing import data_processor
from app.ml.model_selection import model_selector
from app.ml.model_evaluation import model_evaluator
from app.ml.prediction import prediction_service
from app.ml.model_manager import model_manager
from app.ml.monitoring import model_monitor
from app.ml.model_updater import model_updater
from app.utils.exceptions import (
    MLFrameworkError,
    DataProcessingError,
    ModelSelectionError,
    ModelEvaluationError,
    ModelNotFoundError
)
from app.utils.cache import cache_decorator
from app.core.logging_config import get_logger, with_logging_context
from app.db.database import get_db, db_session_decorator
from app.schemas.model import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelInfo, EvaluationRequest, EvaluationResponse,
    ModelUpdate, MonitoringMetrics
)
from app.reporting.report_generator import ReportGenerator

logger = get_logger(__name__)
router = APIRouter()

# متغير عام لتتبع تقدم التدريب
training_progress: Dict[str, Dict[str, Any]] = {}

# Prometheus متريكس
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request duration in seconds', ['endpoint'])
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total number of predictions', ['model_id'])
TRAINING_DURATION = Histogram('model_training_duration_seconds', 'Model training duration in seconds', ['model_id'])
TRAINING_ERRORS = Counter('training_errors_total', 'Training errors', ['error_type'])
FILE_SIZE = Histogram('uploaded_file_size_bytes', 'Size of uploaded files')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')

class NumpyJSONEncoder(json.JSONEncoder):
    """مشفر JSON مخصص للتعامل مع مصفوفات NumPy"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def convert_numpy_types(obj: Any) -> Any:
    """تحويل أنواع NumPy إلى أنواع Python الأساسية"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
@with_logging_context({'operation': 'predict'})
async def predict(
    model_id: str,
    request: PredictionRequest,
    session: AsyncSession = Depends(get_db)
) -> PredictionResponse:
    """إجراء تنبؤات باستخدام النموذج"""
    try:
        # تحميل النموذج
        model = await model_manager.load_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        # إجراء التنبؤات
        predictions = await prediction_service.predict(
            model=model,
            data=request.data,
            return_probabilities=request.return_probabilities
        )

        # تحويل التنبؤات إلى JSON
        predictions = convert_numpy_types(predictions)

        return PredictionResponse(
            model_id=model_id,
            predictions=predictions,
            metadata={'timestamp': datetime.now().isoformat()}
        )

    except Exception as e:
        logger.error(f"خطأ في التنبؤ: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/info", response_model=ModelInfo)
@cache_decorator(ttl=300)
async def get_model_info(
    model_id: str,
    session: AsyncSession = Depends(get_db)
) -> ModelInfo:
    """الحصول على معلومات النموذج"""
    try:
        # استرجاع المعلومات من قاعدة البيانات
        model_info = await model_manager.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        # تحويل البيانات إلى أنواع قابلة للتحويل إلى JSON
        model_info = convert_numpy_types(model_info)

        # التأكد من وجود جميع الحقول المطلوبة
        required_fields = {
            'model_id': model_id,
            'task_type': model_info.get('task_type', 'unknown'),
            'target_column': model_info.get('target_column', ''),
            'feature_names': model_info.get('feature_names', []),
            'creation_date': model_info.get('creation_date', datetime.now().isoformat()),
            'last_updated': model_info.get('last_updated', datetime.now().isoformat()),
            'status': model_info.get('status', 'active'),
            'version': model_info.get('version', '1.0.0'),
            'metadata': {
                'framework_version': settings.app.version,
                'file_name': model_info.get('file_name', ''),
                'training_time': model_info.get('training_time', 0.0),
                'evaluation_results': model_info.get('evaluation_results', {}),
                'training_params': model_info.get('training_params', {})
            }
        }

        # تحديث model_info بالحقول المطلوبة
        model_info.update(required_fields)

        # تسجيل نجاح العملية
        logger.info(f"تم استرجاع معلومات النموذج {model_id} بنجاح")

        # إنشاء نموذج ModelInfo
        return ModelInfo(**model_info)

    except Exception as e:
        logger.error(f"خطأ في الحصول على معلومات النموذج {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل في الحصول على معلومات النموذج: {str(e)}"
        )

@router.post("/models/{model_id}/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    model_id: str,
    request: EvaluationRequest,
    session: AsyncSession = Depends(get_db)
) -> EvaluationResponse:
    """تقييم النموذج"""
    try:
        # تحميل النموذج
        model, metadata = await model_manager.load_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        # قياس وقت التقييم
        evaluation_start = datetime.now()

        # تقييم النموذج
        evaluation_results = await model_evaluator.evaluate_model(
            model=model,
            X=request.data,
            y=request.actual_values,
            task_type=metadata['task_type'],
            feature_names=metadata['feature_names']
        )

        # حساب وقت التقييم
        evaluation_time = (datetime.now() - evaluation_start).total_seconds()

        return EvaluationResponse(
            model_id=model_id,
            evaluation_results=evaluation_results,
            evaluation_time=evaluation_time
        )

    except Exception as e:
        logger.error(f"خطأ في تقييم النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/monitoring", response_model=MonitoringMetrics)
async def get_monitoring_metrics(
    model_id: str,
    session: AsyncSession = Depends(get_db)
) -> MonitoringMetrics:
    """الحصول على مقاييس مراقبة النموذج"""
    try:
        # التحقق من وجود النموذج
        if not await model_manager.model_exists(model_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        metrics = await model_monitor.get_metrics(model_id)
        if not metrics:
            return MonitoringMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                metrics={},
                drift_detected=False,
                performance_metrics={},
                resource_usage={}
            )

        return metrics

    except Exception as e:
        logger.error(f"خطأ في الحصول على مقاييس المراقبة: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/models/{model_id}/update", response_model=ModelUpdate)
async def update_model(
    model_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db)
) -> ModelUpdate:
    """تحديث النموذج"""
    try:
        update_result = await model_updater.update_model(model_id)
        return update_result

    except Exception as e:
        logger.error(f"خطأ في تحديث النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# إنشاء نسخة من ReportGenerator
report_generator = ReportGenerator()

@router.get("/models/{model_id}/report")
async def generate_model_report_endpoint(
    model_id: str,
    session: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    """توليد تقرير النموذج"""
    try:
        # التحقق من وجود النموذج
        model_info = await model_manager.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        # تحويل البيانات إلى JSON
        model_info = convert_numpy_types(model_info)

        # توليد التقرير
        report_path = await report_generator.generate_report(
            model_id=model_id,
            model_info=model_info
        )

        if not report_path or not os.path.exists(report_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في توليد التقرير"
            )

        return StreamingResponse(
            open(report_path, 'rb'),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="model_{model_id}_report.pdf"'
            }
        )

    except Exception as e:
        logger.error(f"خطأ في توليد تقرير النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/list")
async def list_models(
    session: AsyncSession = Depends(get_db),
    status: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    sort_by: Optional[str] = Query('creation_date'),
    order: Optional[str] = Query('desc')
) -> List[ModelInfo]:
    """الحصول على قائمة النماذج مع خيارات التصفية والترتيب"""
    try:
        # الحصول على قائمة النماذج
        models = await model_manager.list_models()
        
        # تحويل كل نموذج إلى ModelInfo
        model_list = []
        for model in models:
            try:
                # تحويل البيانات إلى أنواع قابلة للتحويل إلى JSON
                model_data = convert_numpy_types(model)
                
                # إضافة الحقول المطلوبة
                model_data.update({
                    'model_id': model_data.get('model_id', ''),
                    'task_type': model_data.get('task_type', 'unknown'),
                    'target_column': model_data.get('target_column', ''),
                    'feature_names': model_data.get('feature_names', []),
                    'creation_date': model_data.get('creation_date', datetime.now().isoformat()),
                    'last_updated': model_data.get('last_updated', datetime.now().isoformat()),
                    'status': model_data.get('status', 'active'),
                    'version': model_data.get('version', '1.0.0'),
                    'metadata': {
                        'framework_version': settings.app.version,
                        'file_name': model_data.get('file_name', ''),
                        'training_time': model_data.get('training_time', 0.0)
                    }
                })
                
                model_info = ModelInfo(**model_data)
                model_list.append(model_info)
            except Exception as e:
                logger.error(f"خطأ في تحويل معلومات النموذج: {str(e)}")
                continue
        
        # تطبيق التصفية
        if status:
            model_list = [m for m in model_list if m.status == status]
        if task_type:
            model_list = [m for m in model_list if m.task_type == task_type]
            
        # تطبيق الترتيب
        reverse = order.lower() == 'desc'
        model_list.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
        
        return model_list

    except Exception as e:
        logger.error(f"خطأ في الحصول على قائمة النماذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/models/train", 
             response_model=TrainingResponse,
             status_code=status.HTTP_201_CREATED)
@with_logging_context({'operation': 'train_model'})
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    model_id: Optional[str] = Form(default=None),
    training_params: Optional[str] = Form(default=None),
    session: AsyncSession = Depends(get_db)
) -> TrainingResponse:
    """تدريب نموذج جديد"""
    training_start_time = datetime.now()
    
    try:
        # التحقق من نوع المهمة
        valid_task_types = settings.ml.task_types
        if task_type not in valid_task_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"نوع المهمة غير صالح. الأنواع المدعومة هي: {', '.join(valid_task_types)}"
            )

        # قراءة البيانات
        content = await file.read()
        file_size = len(content)
        FILE_SIZE.observe(file_size)

        # التحقق من حجم الملف
        if file_size > settings.storage.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"حجم الملف يتجاوز الحد الأقصى المسموح به ({settings.storage.max_file_size} بايت)"
            )

        try:
            if file.filename.endswith('.csv'):
                # محاولة قراءة الملف بترميزات مختلفة
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1256', 'iso-8859-1']
                df = None
                last_error = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                        break
                    except Exception as e:
                        last_error = e
                        continue
                        
                if df is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"فشل قراءة ملف CSV: {str(last_error)}"
                    )
                    
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="نوع الملف غير مدعوم. الأنواع المدعومة هي: CSV, Excel"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"فشل قراءة الملف: {str(e)}"
            )

        # التحقق من البيانات
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="البيانات فارغة"
            )

        if target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"عمود الهدف '{target_column}' غير موجود"
            )

        # معالجة معلمات التدريب
        if training_params:
            try:
                params = json.loads(training_params)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="معلمات التدريب غير صالحة. يجب أن تكون بتنسيق JSON"
                )
        else:
            params = {}

        # إنشاء معرف النموذج إذا لم يكن موجوداً
        if not model_id:
            model_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # تحديث حالة التقدم
        training_progress[model_id] = {
            'progress': 0,
            'status': 'جاري تحميل البيانات...'
        }

        try:
            # معالجة البيانات
            training_progress[model_id]['status'] = 'جاري معالجة البيانات...'
            X, y = await data_processor.process_data(df, target_column, task_type)
            training_progress[model_id]['progress'] = 20

            # اختيار وتدريب النموذج
            training_progress[model_id]['status'] = 'جاري تدريب النموذج...'
            model, params = await model_selector.select_best_model(X, y, task_type)
            training_progress[model_id]['progress'] = 60

            # تقييم النموذج
            training_progress[model_id]['status'] = 'جاري تقييم النموذج...'
            evaluation_results = await model_evaluator.evaluate_model(
                model=model,
                X=X,
                y=y,
                task_type=task_type,
                feature_names=df.columns.tolist()
            )
            training_progress[model_id]['progress'] = 80

            # تحويل النتائج إلى أنواع بيانات قابلة للتحويل إلى JSON
            evaluation_results = convert_numpy_types(evaluation_results)

            # حفظ النموذج
            training_progress[model_id]['status'] = 'جاري حفظ النموذج...'
            await model_manager.save_model(
                model=model,
                model_id=model_id,
                metadata={
                    'task_type': task_type,
                    'target_column': target_column,
                    'feature_names': df.columns.tolist(),
                    'training_params': params,
                    'evaluation_results': evaluation_results,
                    'file_name': file.filename,
                    'creation_date': datetime.now().isoformat()
                }
            )
            training_progress[model_id]['progress'] = 100
            training_progress[model_id]['status'] = 'اكتمل التدريب'

            # حساب وقت التدريب
            training_time = (datetime.now() - training_start_time).total_seconds()
            TRAINING_DURATION.labels(model_id=model_id).observe(training_time)

            # إنشاء الاستجابة
            response = TrainingResponse(
                model_id=model_id,
                task_type=task_type,
                target_column=target_column,
                feature_names=df.columns.tolist(),
                parameters=params,
                evaluation_results=evaluation_results,
                training_time=training_time
            )

            return response

        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {str(e)}")
            if model_id in training_progress:
                training_progress[model_id]['status'] = f'فشل التدريب: {str(e)}'
            TRAINING_ERRORS.labels(error_type=type(e).__name__).inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ غير متوقع في تدريب النموذج: {str(e)}")
        TRAINING_ERRORS.labels(error_type='unexpected').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ غير متوقع: {str(e)}"
        )

@router.get("/training-progress/{model_id}")
async def get_training_progress(model_id: str):
    """الحصول على تقدم التدريب"""
    if model_id not in training_progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="معرف النموذج غير موجود"
        )
    return EventSourceResponse(progress_generator(model_id))

async def progress_generator(model_id: str):
    """مولد تحديثات التقدم"""
    while True:
        if model_id in training_progress:
            yield {
                "event": "message",
                "data": json.dumps(training_progress[model_id], cls=NumpyJSONEncoder)
            }
            if training_progress[model_id]['progress'] == 100 or 'error' in training_progress[model_id]['status']:
                break
        await asyncio.sleep(1)

@router.post("/models/{model_id}/stop-training")
async def stop_model_training(model_id: str) -> Dict[str, str]:
    """إيقاف تدريب النموذج"""
    try:
        if model_id not in training_progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"لا يوجد تدريب جارٍ للنموذج {model_id}"
            )
            
        training_progress[model_id]['status'] = 'تم إيقاف التدريب'
        return {"message": f"تم إيقاف تدريب النموذج {model_id}"}

    except Exception as e:
        logger.error(f"خطأ في إيقاف تدريب النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/performance")
async def get_model_performance(
    model_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """الحصول على أداء النموذج خلال فترة زمنية"""
    try:
        performance = await model_monitor.get_performance_metrics(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date
        )
        if not performance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"لا توجد بيانات أداء متحة للنموذج {model_id}"
            )
        return performance

    except Exception as e:
        logger.error(f"خطأ في الحصول على أداء النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/models/{model_id}/retrain")
async def retrain_model(
    model_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, str]:
    """إعادة تدريب النموذج على بيانات جديدة"""
    try:
        # التحقق من وجود النموذج
        model_info = await model_manager.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )

        # إضافة مهمة إعادة التدريب للخلفية
        background_tasks.add_task(
            model_updater.retrain_model,
            model_id=model_id,
            training_file=file,
            original_params=model_info.get('training_params', {})
        )

        return {
            "message": f"تمت إضافة مهمة إعادة تدريب النموذج {model_id} إلى قائمة المهام",
            "status": "pending"
        }

    except Exception as e:
        logger.error(f"خطأ في إعادة تدريب النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/drift")
async def check_model_drift(
    model_id: str,
    window_size: Optional[int] = Query(1000, gt=0)
) -> Dict[str, Any]:
    """فحص انحراف النموذج"""
    try:
        drift_metrics = await model_monitor.check_drift(
            model_id=model_id,
            window_size=window_size
        )
        return drift_metrics

    except Exception as e:
        logger.error(f"خطأ في فحص انحراف النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/health")
async def get_model_health(model_id: str) -> Dict[str, Any]:
    """الحصول على صحة النموذج"""
    try:
        health_metrics = await model_monitor.get_health_metrics(model_id)
        if not health_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"لا توجد معلومات صحة متاحة للنموذج {model_id}"
            )
        return health_metrics

    except Exception as e:
        logger.error(f"خطأ في الحصول على صحة النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/models/compare")
async def compare_models(
    model_ids: List[str],
    metrics: Optional[List[str]] = Query(None),
    dataset: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """مقارنة أداء عدة نماذج"""
    try:
        comparison = await model_evaluator.compare_models(
            model_ids=model_ids,
            metrics=metrics,
            dataset=dataset
        )
        return comparison

    except Exception as e:
        logger.error(f"خطأ في مقارنة النماذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/system/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """الحصول على مقاييس النظام"""
    try:
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': dict(psutil.virtual_memory()._asdict()),
            'disk_usage': dict(psutil.disk_usage('/')._asdict()),
            'models_count': len(await model_manager.list_models()),
            'cache_size': len(model_manager.model_cache),
            'training_jobs': len(training_progress)
        }
        return metrics

    except Exception as e:
        logger.error(f"خطأ في الحصول على مقاييس النظام: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/data/preview")
async def preview_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """معاينة البيانات وتحليلها"""
    try:
        content = await file.read()

        try:
            if file.filename.endswith('.csv'):
                # محاولة قراءة الملف بترميزات مختلفة
                encodings = ['utf-8', 'utf-8-sig',
                             'latin1', 'cp1256', 'iso-8859-1']
                df = None
                last_error = None

                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(content),
                            encoding=encoding,
                            nrows=5,
                            sep=None,
                            engine='python'
                        )
                        break
                    except Exception as e:
                        last_error = e
                        continue

                if df is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"فشل قراءة ملف CSV: {str(last_error)}"
                    )

            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(
                    io.BytesIO(content),
                    engine='openpyxl' if file.filename.endswith(
                        '.xlsx') else 'xlrd',
                    nrows=5
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="نوع الملف غير مدعوم. الأنواع المدعومة هي: CSV, Excel"
                )

            # تحليل البيانات
            analysis = {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
                'sample_data': df.head().to_dict(orient='records'),
                'missing_values': df.isnull().sum().to_dict()
            }

            return analysis

        except Exception as e:
            logger.error(f"خطأ في قراءة الملف: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"فشل قراءة الملف: {str(e)}"
            )

    except Exception as e:
        logger.error(f"خطأ في معاينة البيانات: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )