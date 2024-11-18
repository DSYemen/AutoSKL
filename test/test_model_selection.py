import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from app.ml.model_selection import model_selector

@pytest.fixture
def classification_data():
    """بيانات تصنيف للاختبار"""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def regression_data():
    """بيانات انحدار للاختبار"""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    return X, y

def test_get_models():
    """اختبار الحصول على النماذج"""
    # تصنيف
    model_selector.task_type = 'classification'
    classification_models = model_selector.get_models()
    assert len(classification_models) > 0
    assert all(hasattr(model, 'fit') for model in classification_models.values())
    
    # انحدار
    model_selector.task_type = 'regression'
    regression_models = model_selector.get_models()
    assert len(regression_models) > 0
    assert all(hasattr(model, 'fit') for model in regression_models.values())
    
    # تجميع
    model_selector.task_type = 'clustering'
    clustering_models = model_selector.get_models()
    assert len(clustering_models) > 0
    assert all(hasattr(model, 'fit') for model in clustering_models.values())

def test_get_param_space():
    """اختبار الحصول على نطاق المعلمات"""
    param_space = model_selector.get_param_space('random_forest')
    assert isinstance(param_space, dict)
    assert len(param_space) > 0
    assert all(isinstance(param, tuple) for param in param_space.values())

def test_optimize_model(classification_data):
    """اختبار تحسين النموذج"""
    X, y = classification_data
    model_selector.task_type = 'classification'
    
    model, params, score = model_selector.optimize_model(
        'random_forest',
        X, y,
        n_trials=5  # عدد قليل للاختبار
    )
    
    assert hasattr(model, 'fit')
    assert isinstance(params, dict)
    assert isinstance(score, float)
    assert score > 0

def test_select_best_model(classification_data):
    """اختبار اختيار أفضل نموذج"""
    X, y = classification_data
    
    model, params = model_selector.select_best_model(
        X, y,
        'classification',
        n_trials=5  # عدد قليل للاختبار
    )
    
    assert hasattr(model, 'fit')
    assert isinstance(params, dict)
    assert model_selector.best_model is not None
    assert model_selector.best_params is not None
    assert model_selector.best_score is not None

def test_regression_model_selection(regression_data):
    """اختبار اختيار نموذج الانحدار"""
    X, y = regression_data
    
    model, params = model_selector.select_best_model(
        X, y,
        'regression',
        n_trials=5  # عدد قليل للاختبار
    )
    
    assert hasattr(model, 'fit')
    assert isinstance(params, dict)
    assert model_selector.best_model is not None
    assert model_selector.best_params is not None
    assert model_selector.best_score is not None

def test_invalid_task_type():
    """اختبار نوع مهمة غير صالح"""
    with pytest.raises(Exception):
        model_selector.get_models('invalid_task_type')

def test_model_persistence(classification_data, tmp_path):
    """اختبار استمرارية النموذج"""
    X, y = classification_data
    
    # تدريب واختيار النموذج
    model, params = model_selector.select_best_model(
        X, y,
        'classification',
        n_trials=5
    )
    
    # حفظ النموذج
    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    
    # تحميل النموذج
    loaded_model = joblib.load(model_path)
    
    # التحقق من أن النموذج المحمل يعمل
    assert hasattr(loaded_model, 'predict')
    predictions = loaded_model.predict(X)
    assert len(predictions) == len(y) 