import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.ml.monitoring import model_monitor
from app.db.models import ModelMetrics, DataDrift
from sqlalchemy.orm import Session
from unittest.mock import Mock, patch

@pytest.fixture
def mock_db():
    """قاعدة بيانات وهمية للاختبار"""
    db = Mock(spec=Session)
    return db

@pytest.fixture
def sample_metrics():
    """مقاييس نموذج للاختبار"""
    return {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.82,
        'f1': 0.84
    }

@pytest.fixture
def sample_data():
    """بيانات للاختبار"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': ['A', 'B', 'C'] * 33 + ['A']
    })

async def test_check_model_performance(mock_db, sample_metrics):
    """اختبار التحقق من أداء النموذج"""
    # تجهيز البيانات الوهمية
    historical_metrics = [
        {'metrics': {'accuracy': 0.8}, 'timestamp': datetime.now() - timedelta(days=1)},
        {'metrics': {'accuracy': 0.82}, 'timestamp': datetime.now() - timedelta(days=2)}
    ]
    
    mock_db.query.return_value.filter.return_value.all.return_value = historical_metrics
    
    # اختبار الأداء الجيد
    result = await model_monitor.check_model_performance(
        'test_model',
        sample_metrics,
        mock_db
    )
    assert result is True
    
    # اختبار الأداء المنخفض
    low_metrics = {'accuracy': 0.6}
    result = await model_monitor.check_model_performance(
        'test_model',
        low_metrics,
        mock_db
    )
    assert result is False

async def test_detect_data_drift(mock_db, sample_data):
    """اختبار اكتشاف انحراف البيانات"""
    # بيانات مرجعية
    reference_data = sample_data.copy()
    
    # بيانات جديدة مع انحراف
    current_data = sample_data.copy()
    current_data['feature1'] = current_data['feature1'] + 2  # إضافة انحراف
    
    drift_results = await model_monitor.detect_data_drift(
        'test_model',
        reference_data,
        current_data,
        mock_db
    )
    
    assert isinstance(drift_results, dict)
    assert 'feature1' in drift_results
    assert drift_results['feature1']['drift_detected'] is True
    assert 'feature2' in drift_results
    assert 'feature3' in drift_results

async def test_get_historical_metrics(mock_db):
    """اختبار استرجاع المقاييس التاريخية"""
    metrics = [
        ModelMetrics(
            model_id='test_model',
            timestamp=datetime.now() - timedelta(days=i),
            metrics={'accuracy': 0.8 + i/100}
        )
        for i in range(5)
    ]
    
    mock_db.query.return_value.filter.return_value.all.return_value = metrics
    
    historical_metrics = await model_monitor._get_historical_metrics(
        'test_model',
        mock_db,
        days=7
    )
    
    assert len(historical_metrics) == 5
    assert all('metrics' in m for m in historical_metrics)
    assert all('timestamp' in m for m in historical_metrics)

async def test_save_drift_results(mock_db):
    """اختبار حفظ نتائج الانحراف"""
    await model_monitor._save_drift_results(
        'test_model',
        'feature1',
        True,
        0.85,
        mock_db
    )
    
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

async def test_get_monitoring_summary(mock_db):
    """اختبار الحصول على ملخص المراقبة"""
    # تجهيز البيانات الوهمية
    metrics = [
        ModelMetrics(
            model_id='test_model',
            timestamp=datetime.now(),
            metrics={'accuracy': 0.85}
        )
    ]
    
    drift_records = [
        DataDrift(
            model_id='test_model',
            feature_name='feature1',
            drift_score=0.85,
            drift_detected=True,
            detection_time=datetime.now()
        )
    ]
    
    mock_db.query.return_value.filter.return_value.all.side_effect = [
        metrics, drift_records
    ]
    
    summary = await model_monitor.get_monitoring_summary(
        'test_model',
        mock_db
    )
    
    assert 'model_id' in summary
    assert 'metrics_history' in summary
    assert 'drift_records' in summary
    assert len(summary['metrics_history']) == 1
    assert len(summary['drift_records']) == 1

def test_drift_threshold():
    """اختبار عتبة الانحراف"""
    assert model_monitor.drift_threshold > 0
    assert model_monitor.drift_threshold < 1

def test_performance_threshold():
    """اختبار عتبة الأداء"""
    assert model_monitor.performance_threshold > 0
    assert model_monitor.performance_threshold <= 1 