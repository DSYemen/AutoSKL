from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    """أنواع المهام المدعومة"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class ModelStatus(str, Enum):
    """حالات النموذج"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class PredictionRequest(BaseModel):
    """نموذج طلب التنبؤ"""
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="البيانات المراد التنبؤ بها"
    )
    return_probabilities: bool = Field(
        False,
        description="إرجاع احتمالات التنبؤ"
    )
    batch_size: Optional[int] = Field(
        None,
        description="حجم الدفعة للتنبؤات المتعددة",
        gt=0
    )

    @field_validator('data')
    def validate_data(cls, v: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """التحقق من صحة البيانات"""
        if isinstance(v, dict):
            if not v:
                raise ValueError("البيانات لا يمكن أن تكون فارغة")
        elif isinstance(v, list):
            if not v:
                raise ValueError("قائمة البيانات لا يمكن أن تكون فارغة")
            if not all(isinstance(item, dict) for item in v):
                raise ValueError("جميع العناصر يجب أن تكون قواميس")
        else:
            raise ValueError("البيانات يجب أن تكون قاموس أو قائمة قواميس")
        return v

class PredictionResponse(BaseModel):
    """نموذج استجابة التنبؤ"""
    model_id: str = Field(..., description="معرف النموذج")
    predictions: List[Dict[str, Any]] = Field(..., description="نتائج التنبؤ")
    metadata: Optional[Dict[str, Any]] = Field(None, description="البيانات الوصفية للتنبؤ")
    execution_time: Optional[float] = Field(None, description="وقت التنفيذ بالثواني")

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'model_id': 'model_123',
                    'predictions': [{'class': 1, 'probability': 0.95}],
                    'metadata': {'version': '1.0.0'},
                    'execution_time': 0.15
                }
            ]
        }
    }

class ModelInfo(BaseModel):
    """نموذج معلومات النموذج"""
    model_id: str = Field(..., description="معرف النموذج")
    task_type: TaskType = Field(..., description="نوع المهمة")
    target_column: str = Field(..., description="عمود الهدف")
    status: ModelStatus = Field(default=ModelStatus.ACTIVE, description="حالة النموذج")
    creation_date: datetime = Field(..., description="تاريخ الإنشاء")
    last_updated: Optional[datetime] = Field(None, description="آخر تحديث")
    version: Optional[str] = Field(None, description="إصدار النموذج")
    description: Optional[str] = Field(None, description="وصف النموذج")
    evaluation_results: Dict[str, Any] = Field(..., description="نتائج التقييم")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="أهمية الميزات")
    preprocessing_info: Optional[Dict[str, Any]] = Field(None, description="معلومات المعالجة المسبقة")

    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat()
        }
    }

class TrainingRequest(BaseModel):
    """نموذج طلب التدريب"""
    task_type: TaskType = Field(..., description="نوع المهمة")
    target_column: str = Field(..., description="عمود الهدف")
    model_id: Optional[str] = Field(None, description="معرف النموذج (اختياري)")
    training_params: Optional[Dict[str, Any]] = Field(None, description="معلمات التدريب")
    validation_split: Optional[float] = Field(
        0.2,
        description="نسبة بيانات التحقق",
        ge=0.0,
        le=1.0
    )
    stratify: Optional[bool] = Field(True, description="استخدام التقسيم الطبقي")

class TrainingResponse(BaseModel):
    """نموذج استجابة التدريب"""
    model_id: str
    task_type: str
    target_column: str
    feature_names: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    evaluation_results: Dict[str, Any]
    training_time: float = Field(default=0.0)

class EvaluationRequest(BaseModel):
    """نموذج طلب التقييم"""
    model_id: str = Field(..., description="معرف النموذج")
    target_column: Optional[str] = Field(None, description="عمود الهدف")
    metrics: Optional[List[str]] = Field(None, description="المقاييس المطلوبة")

class EvaluationResponse(BaseModel):
    """نموذج استجابة التقييم"""
    model_id: str = Field(..., description="معرف النموذج")
    evaluation_results: Dict[str, Any] = Field(..., description="نتائج التقييم")
    evaluation_time: float = Field(..., description="وقت التقييم بالثواني")

class MonitoringMetrics(BaseModel):
    """نموذج مقاييس المراقبة"""
    model_id: str = Field(..., description="معرف النموذج")
    timestamp: datetime = Field(..., description="وقت القياس")
    performance_metrics: Dict[str, float] = Field(..., description="مقاييس الأداء")
    drift_metrics: Optional[Dict[str, Any]] = Field(None, description="مقاييس الانحراف")
    resource_usage: Optional[Dict[str, float]] = Field(None, description="استخدام الموارد")

class ModelUpdate(BaseModel):
    """نموذج تحديث النموذج"""
    model_id: str = Field(..., description="معرف النموذج")
    update_type: str = Field(..., description="نوع التحديث")
    performance_before: Dict[str, float] = Field(..., description="الأداء قبل التحديث")
    performance_after: Dict[str, float] = Field(..., description="الأداء بعد التحديث")
    update_time: datetime = Field(..., description="وقت التحديث")
    metadata: Optional[Dict[str, Any]] = Field(None, description="بيانات وصفية إضافية")

    @field_validator('update_type')
    def validate_update_type(cls, v: str) -> str:
        """التحقق من نوع التحديث"""
        allowed_types = {'retrain', 'tune', 'fix'}
        if v not in allowed_types:
            raise ValueError(f"نوع التحديث يجب أن يكون أحد القيم التالية: {allowed_types}")
        return v
  