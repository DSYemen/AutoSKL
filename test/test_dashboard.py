import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from app.visualization.dashboard import dashboard
from app.db.models import Model, ModelUpdate, DataDrift

@pytest.fixture
def mock_db():
    """قاعدة بيانات وهمية للاختبار"""
    db = Mock()
    return db

@pytest.fixture
def sample_models():
    """نماذج للاختبار"""
    return [
        Mock(
            model_id='model1',
            task_type='classification',
            creation_date=datetime.now()
        ),
        Mock(
            model_id='model2',
            task_type='regression',
            creation_date=datetime.now() - timedelta(days=1)
        )
    ]

@pytest.fixture
def sample_updates():
    """تحديثات للاختبار"""
    return [
        Mock(
            model_id='model1',
            update_time=datetime.now() - timedelta(hours=1),
            performance_before={'accuracy': 0.8},
            performance_after={'accuracy': 0.85}
        ),
        Mock(
            model_id='model2',
            update_time=datetime.now() - timedelta(hours=2),
            performance_before={'r2': 0.75},
            performance_after={'r2': 0.78}
        )
    ]

@pytest.mark.asyncio
async def test_get_model_overview(mock_db, sample_models, sample_updates):
    """اختبار الحصول على نظرة عامة على النماذج"""
    mock_db.query.return_value.all.side_effect = [
        sample_models,
        sample_updates
    ]
    
    overview = await dashboard.get_model_overview(mock_db)
    
    assert overview['total_models'] == 2
    assert 'classification' in overview['models_by_type']
    assert 'regression' in overview['models_by_type']
    assert len(overview['recent_updates']) == 2

@pytest.mark.asyncio
async def test_get_model_performance_history(mock_db):
    """اختبار الحصول على تاريخ أداء النموذج"""
    metrics = [
        Mock(
            timestamp=datetime.now() - timedelta(days=i),
            metrics={'accuracy': 0.8 + i/100}
        )
        for i in range(5)
    ]
    
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = metrics
    
    history = await dashboard.get_model_performance_history(
        'test_model',
        mock_db
    )
    
    assert 'plot' in history
    assert 'summary' in history
    assert len(history['summary']) > 0

@pytest.mark.asyncio
async def test_get_drift_analysis(mock_db):
    """اختبار الحصول على تحليل الانحراف"""
    drift_records = [
        Mock(
            detection_time=datetime.now() - timedelta(days=i),
            feature_name=f'feature{i}',
            drift_score=0.1 * i,
            drift_detected=i > 2
        )
        for i in range(5)
    ]
    
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = drift_records
    
    analysis = await dashboard.get_drift_analysis(
        'test_model',
        mock_db
    )
    
    assert 'plot' in analysis
    assert 'summary' in analysis
    assert analysis['summary']['total_checks'] == 5
    assert analysis['summary']['drift_detected_count'] > 0

def test_create_performance_history_plot():
    """اختبار إنشاء مخطط تاريخ الأداء"""
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'accuracy': np.random.rand(10),
        'precision': np.random.rand(10)
    })
    
    fig = dashboard._create_performance_history_plot(df)
    
    assert fig is not None
    assert len(fig.data) == 2  # accuracy and precision traces

def test_create_drift_analysis_plot():
    """اختبار إنشاء مخطط تحليل الانحراف"""
    df = pd.DataFrame({
        'detection_time': pd.date_range(start='2023-01-01', periods=10),
        'feature_name': ['f1', 'f2'] * 5,
        'drift_score': np.random.rand(10)
    })
    
    fig = dashboard._create_drift_analysis_plot(df)
    
    assert fig is not None
    assert len(fig.data) == 2  # two features

def test_calculate_improvement():
    """اختبار حساب نسبة التحسن"""
    before = {'accuracy': 0.8}
    after = {'accuracy': 0.85}
    
    improvement = dashboard._calculate_improvement(before, after)
    
    assert improvement == pytest.approx(6.25)  # (0.85 - 0.8) / 0.8 * 100

def test_calculate_performance_summary():
    """اختبار حساب ملخص الأداء"""
    df = pd.DataFrame({
        'accuracy': [0.8, 0.82, 0.85],
        'precision': [0.75, 0.78, 0.8]
    })
    
    summary = dashboard._calculate_performance_summary(df)
    
    assert 'accuracy' in summary
    assert 'precision' in summary
    assert all(key in summary['accuracy'] for key in ['current', 'mean', 'std', 'trend'])

def test_calculate_drift_summary():
    """اختبار حساب ملخص الانحراف"""
    df = pd.DataFrame({
        'drift_detected': [True, False, True],
        'feature_name': ['f1', 'f2', 'f1'],
        'detection_time': pd.date_range(start='2023-01-01', periods=3)
    })
    
    summary = dashboard._calculate_drift_summary(df)
    
    assert summary['total_checks'] == 3
    assert summary['drift_detected_count'] == 2
    assert summary['features_affected'] == 1 