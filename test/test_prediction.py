import pytest
import numpy as np
import pandas as pd
from app.ml.prediction import prediction_service
from app.ml.data_processing import data_processor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

@pytest.fixture
def sample_classification_model():
    """نموذج تصنيف للاختبار"""
    model = RandomForestClassifier(random_state=42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model, X, y

@pytest.fixture
def sample_regression_model():
    """نموذج انحدار للاختبار"""
    model = RandomForestRegressor(random_state=42)
    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    model.fit(X, y)
    return model, X, y

def test_setup():
    """اختبار إعداد خدمة التنبؤ"""
    model = RandomForestClassifier()
    feature_names = ['f1', 'f2', 'f3', 'f4']
    
    prediction_service.setup(
        model=model,
        task_type='classification',
        feature_names=feature_names
    )
    
    assert prediction_service.model == model
    assert prediction_service.task_type == 'classification'
    assert prediction_service.feature_names == feature_names

def test_validate_input():
    """اختبار التحقق من صحة المدخلات"""
    prediction_service.setup(
        model=RandomForestClassifier(),
        task_type='classification',
        feature_names=['f1', 'f2']
    )
    
    # اختبار قاموس صالح
    valid_dict = {'f1': 1.0, 'f2': 2.0}
    df = prediction_service.validate_input(valid_dict)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['f1', 'f2']
    
    # اختبار DataFrame صالح
    valid_df = pd.DataFrame({'f1': [1.0], 'f2': [2.0]})
    df = prediction_service.validate_input(valid_df)
    assert isinstance(df, pd.DataFrame)
    
    # اختبار مدخلات غير صالحة
    with pytest.raises(Exception):
        prediction_service.validate_input({'invalid_feature': 1.0})

def test_format_prediction():
    """اختبار تنسيق التنبؤات"""
    predictions = np.array([0, 1, 0])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    
    # تنسيق مع الاحتمالات
    results = prediction_service.format_prediction(predictions, probabilities)
    assert len(results) == 3
    assert all('prediction' in r for r in results)
    assert all('probabilities' in r for r in results)
    
    # تنسيق بدون احتمالات
    results = prediction_service.format_prediction(predictions)
    assert len(results) == 3
    assert all('prediction' in r for r in results)
    assert all('probabilities' not in r for r in results)

def test_predict_classification(sample_classification_model):
    """اختبار التنبؤ للتصنيف"""
    model, X, _ = sample_classification_model
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    
    prediction_service.setup(
        model=model,
        task_type='classification',
        feature_names=feature_names
    )
    
    # تنبؤ واحد
    single_input = {f'f{i}': X[0, i] for i in range(X.shape[1])}
    result = prediction_service.predict(single_input)
    assert isinstance(result, list)
    assert len(result) == 1
    
    # تنبؤات متعددة
    multi_input = pd.DataFrame(X, columns=feature_names)
    results = prediction_service.predict(multi_input)
    assert isinstance(results, list)
    assert len(results) == len(X)

def test_predict_regression(sample_regression_model):
    """اختبار التنبؤ للانحدار"""
    model, X, _ = sample_regression_model
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    
    prediction_service.setup(
        model=model,
        task_type='regression',
        feature_names=feature_names
    )
    
    # تنبؤ واحد
    single_input = {f'f{i}': X[0, i] for i in range(X.shape[1])}
    result = prediction_service.predict(single_input)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0]['prediction'], float)

def test_predict_batch():
    """اختبار التنبؤ للدفعات"""
    model = RandomForestClassifier(random_state=42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    prediction_service.setup(
        model=model,
        task_type='classification',
        feature_names=feature_names
    )
    
    # إنشاء بيانات الدفعة
    batch_data = [
        {f'f{i}': x[i] for i in range(len(x))}
        for x in X[:10]
    ]
    
    results = prediction_service.predict_batch(batch_data, batch_size=5)
    assert isinstance(results, list)
    assert len(results) == 10 