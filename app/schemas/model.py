from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import numpy as np


class ModelStatus(str, Enum):
    """حالات النموذج"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class BaseModelConfig(BaseModel):
    """النموذج الأساسي مع الإعدادات"""
    model_config = ConfigDict(protected_namespaces=())


class PredictionRequest(BaseModelConfig):
    """نموذج طلب التنبؤ"""
    data: Union[List[List[float]], List[Dict[str, Any]]]
    return_probabilities: bool = False

    @field_validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("البيانات لا يمكن أن تكون فارغة")
        return v


class PredictionResponse(BaseModelConfig):
    """نموذج استجابة التنبؤ"""
    model_id: str
    predictions: List[Any]
    prediction_time: Optional[float] = None
    metadata: Dict[str, Any]


class TrainingRequest(BaseModelConfig):
    """نموذج طلب التدريب"""
    model_id: Optional[str] = None
    task_type: str
    target_column: str
    feature_names: List[str]
    training_params: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModelConfig):
    """نموذج استجابة التدريب"""
    model_id: str
    task_type: str
    target_column: str
    feature_names: List[str]
    parameters: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    training_time: float


class ModelInfo(BaseModelConfig):
    """نموذج معلومات النموذج"""
    model_id: str
    task_type: str
    target_column: str
    feature_names: List[str]
    creation_date: datetime
    last_updated: Optional[datetime]
    status: ModelStatus
    version: Optional[str]
    metadata: Dict[str, Any]


class EvaluationRequest(BaseModelConfig):
    """نموذج طلب التقييم"""
    data: Union[List[List[float]], List[Dict[str, Any]]]
    actual_values: List[Any]
    metrics: Optional[List[str]] = None


class EvaluationResponse(BaseModelConfig):
    """نموذج استجابة التقييم"""
    model_id: str
    evaluation_results: Dict[str, Any]
    evaluation_time: float


class ModelUpdate(BaseModelConfig):
    """نموذج تحديث النموذج"""
    model_id: str
    update_type: str
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    update_time: datetime
    success: bool
    error_message: Optional[str] = None


class MonitoringMetrics(BaseModelConfig):
    """نموذج مقاييس المراقبة"""
    model_id: str
    timestamp: datetime
    metrics: Dict[str, Any]
    drift_detected: bool
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
