from typing import Optional, Dict, Any, NoReturn
from fastapi import HTTPException, status
import logging
from datetime import datetime
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ErrorDetails: TypeAlias = Dict[str, Any]
ErrorResponse: TypeAlias = Dict[str, Any]

class MLFrameworkError(Exception):
    """الفئة الأساسية للاستثناءات في إطار العمل"""
    def __init__(
        self, 
        message: str,
        error_code: Optional[str] = None,
        details: Optional[ErrorDetails] = None
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.error_code: str = error_code or self.__class__.__name__
        self.details: ErrorDetails = details or {}
        self.timestamp: datetime = datetime.utcnow()
        
        # تسجيل الخطأ
        logger.error(
            f"Error {self.error_code}: {self.message}",
            extra={
                'error_details': self.details,
                'timestamp': self.timestamp
            }
        )

class DataProcessingError(MLFrameworkError):
    """استثناء معالجة البيانات"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            details=details
        )

class ModelSelectionError(MLFrameworkError):
    """استثناء اختيار النموذج"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_SELECTION_ERROR",
            details=details
        )

class ModelEvaluationError(MLFrameworkError):
    """استثناء تقييم النموذج"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_EVALUATION_ERROR",
            details=details
        )

class PredictionError(MLFrameworkError):
    """استثناء التنبؤ"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details
        )

class ModelNotFoundError(MLFrameworkError):
    """استثناء عدم وجود النموذج"""
    def __init__(self, model_id: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=f"النموذج {model_id} غير موجود",
            error_code="MODEL_NOT_FOUND",
            details=details
        )

class ValidationError(MLFrameworkError):
    """استثناء التحقق من الصحة"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class DatabaseError(MLFrameworkError):
    """استثناء قاعدة البيانات"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )

class CacheError(MLFrameworkError):
    """استثناء التخزين المؤقت"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details
        )

class ConfigurationError(MLFrameworkError):
    """استثناء التكوين"""
    def __init__(self, message: str, details: Optional[ErrorDetails] = None) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )

def http_error_handler(error: MLFrameworkError) -> HTTPException:
    """تحويل استثناءات إطار العمل إلى استثناءات HTTP"""
    error_mapping = {
        DataProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ModelSelectionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ModelEvaluationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        PredictionError: status.HTTP_400_BAD_REQUEST,
        ModelNotFoundError: status.HTTP_404_NOT_FOUND,
        ValidationError: status.HTTP_400_BAD_REQUEST,
        DatabaseError: status.HTTP_503_SERVICE_UNAVAILABLE,
        CacheError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR
    }
    
    status_code = error_mapping.get(type(error), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return HTTPException(
        status_code=status_code,
        detail={
            'error_code': error.error_code,
            'message': str(error),
            'details': error.details,
            'timestamp': error.timestamp.isoformat()
        }
    )

def format_error_response(error: Exception) -> ErrorResponse:
    """تنسيق استجابة الخطأ"""
    if isinstance(error, MLFrameworkError):
        return {
            'error_code': error.error_code,
            'message': str(error),
            'details': error.details,
            'timestamp': error.timestamp.isoformat()
        }
    else:
        return {
            'error_code': 'UNKNOWN_ERROR',
            'message': str(error),
            'details': {},
            'timestamp': datetime.utcnow().isoformat()
        } 