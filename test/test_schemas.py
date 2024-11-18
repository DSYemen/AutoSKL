import pytest
from datetime import datetime
from pydantic import ValidationError
from app.schemas.model import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    TrainingResponse,
    EvaluationResponse,
    FeatureImportance,
    ModelMetrics,
    ModelParameters
)

@pytest.fixture
def sample_prediction_data():
    """بيانات تنبؤ نموذجية"""
    return {
        'data': {
            'feature1': 1.0,
            'feature2': 2.0
        },
        'return_probabilities': True
    }

@pytest.fixture
def sample_model_info():
    """معلومات نموذج نموذجية"""
    return {
        'model_id': 'test_model',
        'task_type': 'classification',
        'target_column': 'target',
        'creation_date': datetime.now(),
        'evaluation_results': {
            'accuracy': 0.85,
            'precision': 0.83
        }
    }

def test_prediction_request(sample_prediction_data):
    """اختبار مخطط طلب التنبؤ"""
    request = PredictionRequest(**sample_prediction_data)
    assert isinstance(request.data, dict)
    assert request.return_probabilities is True

def test_prediction_response():
    """اختبار مخطط استجابة التنبؤ"""
    data = {
        'model_id': 'test_model',
        'predictions': [
            {'prediction': 1, 'probabilities': [0.2, 0.8]}
        ]
    }
    response = PredictionResponse(**data)
    assert response.model_id == 'test_model'
    assert len(response.predictions) == 1

def test_model_info(sample_model_info):
    """اختبار مخطط معلومات النموذج"""
    info = ModelInfo(**sample_model_info)
    assert info.model_id == 'test_model'
    assert info.task_type == 'classification'
    assert isinstance(info.creation_date, datetime)

def test_training_response():
    """اختبار مخطط استجابة التدريب"""
    data = {
        'model_id': 'test_model',
        'task_type': 'classification',
        'parameters': {'n_estimators': 100},
        'evaluation_results': {'accuracy': 0.85}
    }
    response = TrainingResponse(**data)
    assert response.model_id == 'test_model'
    assert 'n_estimators' in response.parameters

def test_evaluation_response():
    """اختبار مخطط استجابة التقييم"""
    data = {
        'model_id': 'test_model',
        'evaluation_results': {
            'accuracy': 0.85,
            'precision': 0.83
        }
    }
    response = EvaluationResponse(**data)
    assert 'accuracy' in response.evaluation_results

def test_feature_importance():
    """اختبار مخطط أهمية الميزات"""
    data = {
        'feature_name': 'feature1',
        'importance_score': 0.75
    }
    importance = FeatureImportance(**data)
    assert importance.feature_name == 'feature1'
    assert importance.importance_score == 0.75

def test_model_metrics():
    """اختبار مخطط مقاييس النموذج"""
    data = {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.82,
        'f1_score': 0.84
    }
    metrics = ModelMetrics(**data)
    assert metrics.accuracy == 0.85
    assert metrics.precision == 0.83

def test_model_parameters():
    """اختبار مخطط معلمات النموذج"""
    data = {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'hyperparameters': {'n_estimators': 100},
        'feature_preprocessing': {'scaling': True}
    }
    params = ModelParameters(**data)
    assert params.model_type == 'random_forest'
    assert 'n_estimators' in params.hyperparameters

def test_invalid_prediction_request():
    """اختبار طلب تنبؤ غير صالح"""
    with pytest.raises(ValidationError):
        PredictionRequest(data=123)  # يجب أن يكون قاموساً

def test_invalid_model_info():
    """اختبار معلومات نموذج غير صالحة"""
    with pytest.raises(ValidationError):
        ModelInfo(
            model_id='test_model',
            task_type='invalid_task',  # نوع مهمة غير صالح
            target_column='target',
            evaluation_results={}
        )

def test_optional_fields():
    """اختبار الحقول الاختيارية"""
    # ModelMetrics مع حقول اختيارية فقط
    metrics = ModelMetrics()
    assert metrics.accuracy is None
    assert metrics.precision is None
    
    # ModelInfo بدون تاريخ إنشاء
    info = ModelInfo(
        model_id='test_model',
        task_type='classification',
        target_column='target',
        evaluation_results={}
    )
    assert info.creation_date is None 