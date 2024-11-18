import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from app.reporting.report_generator import report_generator
from unittest.mock import patch, Mock

@pytest.fixture
def sample_model_info():
    """معلومات نموذج للاختبار"""
    return {
        'task_type': 'classification',
        'target_column': 'target',
        'creation_date': datetime.now().isoformat()
    }

@pytest.fixture
def sample_evaluation_results():
    """نتائج تقييم للاختبار"""
    return {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.82,
        'f1': 0.84,
        'confusion_matrix': [[40, 10], [5, 45]],
        'classification_report': {
            'accuracy': 0.85,
            'macro avg': {'precision': 0.83, 'recall': 0.82, 'f1-score': 0.84}
        }
    }

@pytest.fixture
def sample_feature_importance():
    """أهمية الميزات للاختبار"""
    return {
        'feature1': 0.3,
        'feature2': 0.5,
        'feature3': 0.2
    }

@pytest.fixture
def sample_monitoring_data():
    """بيانات مراقبة للاختبار"""
    return {
        'alerts': [
            {
                'message': 'انخفاض في الأداء',
                'timestamp': datetime.now().isoformat()
            }
        ],
        'performance_metrics': {
            'accuracy': {
                'current': 0.85,
                'baseline': 0.82,
                'change': 0.03
            }
        },
        'drift_metrics': {
            'feature1': {
                'drift_score': 0.1,
                'p_value': 0.05,
                'drift_detected': True
            }
        },
        'prediction_metrics': {
            'total_predictions': 1000,
            'average_confidence': 0.85
        }
    }

async def test_generate_model_report(
    sample_model_info,
    sample_evaluation_results,
    sample_feature_importance,
    tmp_path
):
    """اختبار توليد تقرير النموذج"""
    with patch('app.reporting.report_generator.Path') as mock_path:
        mock_path.return_value = tmp_path
        
        report_path = await report_generator.generate_model_report(
            'test_model',
            sample_model_info,
            sample_evaluation_results,
            sample_feature_importance
        )
        
        assert isinstance(report_path, str)
        assert report_path.endswith('.pdf')
        assert Path(report_path).exists()

async def test_generate_monitoring_report(
    sample_monitoring_data,
    tmp_path
):
    """اختبار توليد تقرير المراقبة"""
    with patch('app.reporting.report_generator.Path') as mock_path:
        mock_path.return_value = tmp_path
        
        report_path = await report_generator.generate_monitoring_report(
            'test_model',
            sample_monitoring_data
        )
        
        assert isinstance(report_path, str)
        assert report_path.endswith('.pdf')
        assert Path(report_path).exists()

def test_create_report_plots(
    sample_evaluation_results,
    sample_feature_importance
):
    """اختبار إنشاء مخططات التقرير"""
    report_data = {
        'model_id': 'test_model',
        'evaluation_results': sample_evaluation_results,
        'feature_importance': sample_feature_importance,
        'predictions_history': [
            {
                'timestamp': datetime.now(),
                'accuracy': 0.85,
                'predictions_count': 100
            }
        ]
    }
    
    plots = report_generator._create_report_plots(report_data)
    
    assert isinstance(plots, dict)
    assert 'feature_importance' in plots
    assert 'confusion_matrix' in plots
    assert 'performance_history' in plots

def test_create_performance_history_plot():
    """اختبار إنشاء مخطط تاريخ الأداء"""
    history = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10),
        'accuracy': np.random.rand(10),
        'predictions_count': np.random.randint(50, 150, 10)
    })
    
    fig = report_generator._create_performance_history_plot(history)
    
    assert fig is not None
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0

def test_save_plot(tmp_path):
    """اختبار حفظ المخطط"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
    
    with patch('app.reporting.report_generator.Path') as mock_path:
        mock_path.return_value = tmp_path
        
        output_path = report_generator._save_plot(fig, 'test_plot.png')
        
        assert isinstance(output_path, str)
        assert output_path.endswith('.png')
        assert Path(output_path).exists()

def test_template_rendering(
    sample_model_info,
    sample_evaluation_results
):
    """اختبار تقديم القالب"""
    template = report_generator.template_env.get_template('model_report.html')
    
    html_content = template.render(
        model_id='test_model',
        model_info=sample_model_info,
        evaluation_results=sample_evaluation_results,
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    assert isinstance(html_content, str)
    assert 'test_model' in html_content
    assert str(sample_evaluation_results['accuracy']) in html_content 