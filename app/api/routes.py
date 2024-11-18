from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Response, BackgroundTasks, Query, status
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
from sse_starlette.sse import EventSourceResponse
from prometheus_client import Counter, Histogram, Summary
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
import io

from app.ml.data_processing import data_processor
from app.ml.model_selection import model_selector
from app.ml.model_evaluation import model_evaluator
from app.ml.prediction import prediction_service
from app.ml.model_manager import model_manager
from app.ml.monitoring import model_monitor
from app.utils.exceptions import MLFrameworkError, http_error_handler
from app.utils.cache import cache_decorator
from app.core.logging_config import get_logger, with_logging_context
from app.db.database import get_db, db_session_decorator
from app.schemas.model import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelInfo, EvaluationRequest, EvaluationResponse,
    ModelUpdate, MonitoringMetrics
)

logger = get_logger(__name__)
router = APIRouter()

# متغير عام لتتبع تقدم التدريب
training_progress: Dict[str, Dict[str, Any]] = {}

# Prometheus متريكس
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request duration in seconds', ['endpoint'])
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total number of predictions', ['model_id'])
TRAINING_DURATION = Histogram('model_training_duration_seconds', 'Model training duration in seconds', ['model_id'])

@router.post("/data/preview")
async def preview_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """معاينة البيانات وتحليلها"""
    try:
        content = await file.read()
        
        try:
            if file.filename.endswith('.csv'):
                # محاولة قراءة الملف بترميزات مختلفة
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1256', 'iso-8859-1']
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
                    engine='openpyxl' if file.filename.endswith('.xlsx') else 'xlrd',
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
    try:
        logger.info(f"بدء طلب تدريب نموذج جديد: task_type={task_type}, target_column={target_column}")
        
        # قراءة البيانات
        content = await file.read()
        df = None
        
        try:
            if file.filename.endswith('.csv'):
                # محاولة قراءة الملف بترميزات مختلفة
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1256', 'iso-8859-1']
                last_error = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                        logger.info(f"تم قراءة الملف CSV باستخدام الترميز: {encoding}")
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
                logger.info("تم قراءة ملف Excel بنجاح")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="نوع الملف غير مدعوم. الأنواع المدعومة هي: CSV, Excel"
                )
        except Exception as e:
            logger.error(f"خطأ في قراءة الملف: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"فشل قراءة الملف: {str(e)}"
            )

        # التحقق من البيانات
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="البيانات فارغة"
            )

        if target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"عمود الهدف '{target_column}' غير موجود. الأعمدة المتوفرة: {', '.join(df.columns.tolist())}"
            )

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
                model, X, y, task_type, df.columns.tolist()
            )
            training_progress[model_id]['progress'] = 80

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

            logger.info(f"تم تدريب النموذج بنجاح: {model_id}")
            return TrainingResponse(
                model_id=model_id,
                task_type=task_type,
                parameters=params,
                evaluation_results=evaluation_results,
                training_time=0.0,  # يمكن إضافة حساب وقت التدريب
                feature_importance=evaluation_results.get('feature_importance')
            )

        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {str(e)}")
            if model_id in training_progress:
                training_progress[model_id]['status'] = f'فشل التدريب: {str(e)}'
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ غير متوقع في تدريب النموذج: {str(e)}")
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
                "data": json.dumps(training_progress[model_id])
            }
            if training_progress[model_id]['progress'] == 100 or 'error' in training_progress[model_id]['status']:
                break
        await asyncio.sleep(1)

@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
@with_logging_context({'operation': 'predict'})
@REQUEST_LATENCY.time()
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

        # تحديث العداد
        MODEL_PREDICTIONS.labels(model_id=model_id).inc()

        return PredictionResponse(
            model_id=model_id,
            predictions=predictions,
            metadata={'version': getattr(model, 'version', 'unknown')}
        )

    except Exception as e:
        logger.error(f"خطأ في التنبؤ: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/info", response_model=ModelInfo)
@cache_decorator(ttl=300)  # تخزين مؤقت لمدة 5 دقائق
async def get_model_info(
    model_id: str,
    session: AsyncSession = Depends(get_db)
) -> ModelInfo:
    """الحصول على معلومات النموذج"""
    try:
        model_info = await model_manager.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"النموذج {model_id} غير موجود"
            )
        return model_info

    except Exception as e:
        logger.error(f"خطأ في الحصول على معلومات النموذج: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/models/{model_id}/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    model_id: str,
    request: EvaluationRequest,
    session: AsyncSession = Depends(get_db)
) -> EvaluationResponse:
    """تقييم النموذج"""
    try:
        evaluation_results = await model_evaluator.evaluate_model(
            model_id=model_id,
            metrics=request.metrics
        )
        return EvaluationResponse(
            model_id=model_id,
            evaluation_results=evaluation_results,
            evaluation_time=0.0  # يمكن إضافة حساب وقت التقييم
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
        metrics = await model_monitor.get_metrics(model_id)
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"لا توجد مقاييس متاحة للنموذج {model_id}"
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