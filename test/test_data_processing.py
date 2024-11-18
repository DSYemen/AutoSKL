import pytest
import pandas as pd
import numpy as np
from app.ml.data_processing import data_processor

@pytest.fixture
def sample_data():
    """بيانات اختبار"""
    return pd.DataFrame({
        'numeric_1': [1.0, 2.0, 3.0, np.nan, 5.0],
        'numeric_2': [10, 20, 30, 40, 50],
        'categorical_1': ['A', 'B', 'A', None, 'C'],
        'categorical_2': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [0, 1, 0, 1, 1]
    })

def test_infer_task_type():
    """اختبار استنتاج نوع المهمة"""
    # تصنيف
    y_class = pd.Series([0, 1, 0, 1, 1])
    assert data_processor.infer_task_type(y_class) == 'classification'
    
    # انحدار
    y_reg = pd.Series([1.5, 2.7, 3.2, 4.1, 5.9])
    assert data_processor.infer_task_type(y_reg) == 'regression'

def test_identify_features(sample_data):
    """اختبار تحديد أنواع الميزات"""
    data_processor.identify_features(sample_data)
    
    assert set(data_processor.numeric_features) == {'numeric_1', 'numeric_2'}
    assert set(data_processor.categorical_features) == {'categorical_1', 'categorical_2'}

def test_process_data(sample_data):
    """اختبار معالجة البيانات"""
    X, y = data_processor.process_data(
        sample_data,
        target_column='target',
        task_type='classification'
    )
    
    # التحقق من شكل المخرجات
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    
    # التحقق من معالجة القيم المفقودة
    assert not np.isnan(X).any()

def test_transform_new_data(sample_data):
    """اختبار تحويل بيانات جديدة"""
    # تجهيز المعالج
    data_processor.process_data(
        sample_data,
        target_column='target',
        task_type='classification'
    )
    
    # بيانات جديدة للاختبار
    new_data = pd.DataFrame({
        'numeric_1': [1.5, np.nan],
        'numeric_2': [15, 25],
        'categorical_1': ['B', 'D'],
        'categorical_2': ['X', 'Y']
    })
    
    X_new = data_processor.transform_new_data(new_data)
    
    # التحقق من المخرجات
    assert isinstance(X_new, np.ndarray)
    assert len(X_new) == 2
    assert not np.isnan(X_new).any()

def test_get_feature_names(sample_data):
    """اختبار الحصول على أسماء الميزات"""
    data_processor.process_data(
        sample_data,
        target_column='target',
        task_type='classification'
    )
    
    feature_names = data_processor.get_feature_names()
    
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert all(isinstance(name, str) for name in feature_names) 