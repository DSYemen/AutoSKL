import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from app.ml.model_evaluation import model_evaluator

@pytest.fixture
def classification_data():
    """بيانات تصنيف للاختبار"""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X, y

@pytest.fixture
def regression_data():
    """بيانات انحدار للاختبار"""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model, X, y

@pytest.fixture
def clustering_data():
    """بيانات تجميع للاختبار"""
    X, _ = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        random_state=42
    )
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    return model, X

def test_evaluate_classification(classification_data):
    """اختبار تقييم نموذج التصنيف"""
    model, X, y = classification_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    results = model_evaluator.evaluate_model(
        model, X, y,
        'classification',
        feature_names
    )
    
    assert 'accuracy' in results
    assert 'precision_macro' in results
    assert 'recall_macro' in results
    assert 'f1_macro' in results
    assert 'confusion_matrix' in results
    assert 'classification_report' in results
    assert 'feature_importance' in results
    assert isinstance(results['confusion_matrix'], list)

def test_evaluate_regression(regression_data):
    """اختبار تقييم نموذج الانحدار"""
    model, X, y = regression_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    results = model_evaluator.evaluate_model(
        model, X, y,
        'regression',
        feature_names
    )
    
    assert 'mse' in results
    assert 'rmse' in results
    assert 'mae' in results
    assert 'r2' in results
    assert 'feature_importance' in results
    assert all(isinstance(v, float) for k, v in results.items() if k != 'feature_importance')

def test_evaluate_clustering(clustering_data):
    """اختبار تقييم نموذج التجميع"""
    model, X = clustering_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    results = model_evaluator.evaluate_model(
        model, X, None,
        'clustering',
        feature_names
    )
    
    assert 'silhouette' in results
    assert 'calinski_harabasz' in results
    assert all(isinstance(v, float) for v in results.values())

def test_calculate_feature_importance(classification_data):
    """اختبار حساب أهمية الميزات"""
    model, X, _ = classification_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    importance = model_evaluator.calculate_feature_importance(
        model, X, feature_names
    )
    
    assert isinstance(importance, dict)
    assert len(importance) == len(feature_names)
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())

def test_calculate_shap_values(classification_data):
    """اختبار حساب قيم SHAP"""
    model, X, _ = classification_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    model_evaluator.task_type = 'classification'
    shap_dict = model_evaluator.calculate_shap_values(
        model, X, feature_names
    )
    
    assert 'values' in shap_dict
    assert 'feature_names' in shap_dict
    assert isinstance(shap_dict['values'], np.ndarray)
    assert len(shap_dict['feature_names']) == X.shape[1]

def test_invalid_task_type(classification_data):
    """اختبار نوع مهمة غير صالح"""
    model, X, y = classification_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    with pytest.raises(Exception):
        model_evaluator.evaluate_model(
            model, X, y,
            'invalid_task',
            feature_names
        )

def test_missing_target(classification_data):
    """اختبار هدف مفقود للتصنيف"""
    model, X, _ = classification_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    with pytest.raises(Exception):
        model_evaluator.evaluate_model(
            model, X, None,
            'classification',
            feature_names
        ) 