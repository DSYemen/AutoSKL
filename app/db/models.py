from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text, Enum
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime
import enum
from app.db.database import Base
from typing import Dict, Any, Optional, List
import json

class ModelStatus(str, enum.Enum):
    """حالات النموذج"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class TaskType(str, enum.Enum):
    """أنواع المهام"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class Model(Base):
    """نموذج لتخزين معلومات النماذج"""
    __tablename__ = "models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType), nullable=False)
    target_column: Mapped[str] = mapped_column(String(100), nullable=False)
    creation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_updated: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.ACTIVE)
    version: Mapped[Optional[str]] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text)
    model_info: Mapped[Dict[str, Any]] = mapped_column(JSON)
    
    # العلاقات
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")
    metrics: Mapped[List["ModelMetrics"]] = relationship("ModelMetrics", back_populates="model", cascade="all, delete-orphan")
    updates: Mapped[List["ModelUpdate"]] = relationship("ModelUpdate", back_populates="model", cascade="all, delete-orphan")
    drift_checks: Mapped[List["DataDrift"]] = relationship("DataDrift", back_populates="model", cascade="all, delete-orphan")
    feature_importance: Mapped[List["FeatureImportance"]] = relationship("FeatureImportance", back_populates="model", cascade="all, delete-orphan")
    
    @hybrid_property
    def is_active(self) -> bool:
        return self.status == ModelStatus.ACTIVE
    
    @hybrid_property
    def age_days(self) -> int:
        return (datetime.utcnow() - self.creation_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل النموذج إلى قاموس"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'task_type': self.task_type.value,
            'target_column': self.target_column,
            'creation_date': self.creation_date.isoformat(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'status': self.status.value,
            'version': self.version,
            'description': self.description,
            'model_info': self.model_info,
            'age_days': self.age_days
        }

class Prediction(Base):
    """نموذج لتخزين سجلات التنبؤات"""
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[str] = mapped_column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    input_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    prediction: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    actual_value: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    processing_time: Mapped[Optional[float]] = mapped_column(Float)  # بالثواني
    prediction_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # العلاقات
    model: Mapped["Model"] = relationship("Model", back_populates="predictions")
    
    @hybrid_property
    def is_correct(self) -> Optional[bool]:
        """التحقق من صحة التنبؤ"""
        if self.actual_value is None:
            return None
        return self.prediction == self.actual_value

class ModelMetrics(Base):
    """نموذج لتخزين مقاييس النموذج"""
    __tablename__ = "model_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[str] = mapped_column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset_size: Mapped[Optional[int]] = mapped_column(Integer)
    dataset_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    computation_time: Mapped[Optional[float]] = mapped_column(Float)  # بالثواني
    
    # العلاقات
    model: Mapped["Model"] = relationship("Model", back_populates="metrics")
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """الحصول على قيمة مقياس محدد"""
        return self.metrics.get(metric_name)

class ModelUpdate(Base):
    """نموذج لتخزين تحديثات النموذج"""
    __tablename__ = "model_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    update_time = Column(DateTime(timezone=True), server_default=func.now())
    update_type = Column(String(50))  # e.g., 'retrain', 'tune', 'fix'
    performance_before = Column(JSON)
    performance_after = Column(JSON)
    update_info = Column(JSON)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # العلاقات
    model = relationship("Model", back_populates="updates")
    
    @hybrid_property
    def performance_improvement(self) -> float:
        """حساب نسبة تحسن الأداء"""
        if not (self.performance_before and self.performance_after):
            return 0.0
        before = self.performance_before.get('accuracy', self.performance_before.get('r2', 0))
        after = self.performance_after.get('accuracy', self.performance_after.get('r2', 0))
        return ((after - before) / before) * 100 if before != 0 else 0.0

class DataDrift(Base):
    """نموذج لتخزين معلومات انحراف البيانات"""
    __tablename__ = "data_drift"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    detection_time = Column(DateTime(timezone=True), server_default=func.now())
    feature_name = Column(String(100))
    drift_score = Column(Float)
    drift_detected = Column(Boolean)
    drift_type = Column(String(50))  # e.g., 'concept_drift', 'feature_drift'
    severity = Column(String(20))  # 'low', 'medium', 'high'
    drift_info = Column(JSON)
    
    # العلاقات
    model = relationship("Model", back_populates="drift_checks")

class FeatureImportance(Base):
    """نموذج لتخزين أهمية الميزات"""
    __tablename__ = "feature_importance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    feature_name = Column(String(100))
    importance_score = Column(Float)
    importance_type = Column(String(50))  # e.g., 'shap', 'permutation', 'gain'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    feature_info = Column(JSON)
    
    # العلاقات
    model = relationship("Model", back_populates="feature_importance")

class TrainingJob(Base):
    """نموذج لتخزين معلومات مهام التدريب"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), ForeignKey("models.model_id", ondelete="CASCADE"))
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(String(20))  # 'pending', 'running', 'completed', 'failed'
    error_message = Column(Text)
    training_params = Column(JSON)
    resource_usage = Column(JSON)  # CPU, memory, etc.
    job_info = Column(JSON)
    
    @hybrid_property
    def duration_seconds(self) -> float:
        """حساب مدة التدريب بالثواني"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0