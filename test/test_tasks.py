import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from app.utils.tasks import (
    celery_app,
    update_models_task,
    monitor_models_task,
    generate_reports_task,
    retrain_model_task
)
from app.db.models import Model
from app.core.config import settings

@pytest.fixture
def mock_db():
    """قاعدة بيانات وهمية للاختبار"""
    db = Mock()
    db.query.return_value.all.return_value = [
        Mock(model_id='test_model_1'),
        Mock(model_id='test_model_2')
    ]
    return db

@pytest.fixture
def mock_model_updater():
    """محدث نماذج وهمي للاختبار"""
    with patch('app.utils.tasks.model_updater') as mock:
        mock.check_and_update_models.return_value = {
            'test_model_1': True,
            'test_model_2': False
        }
        yield mock

@pytest.fixture
def mock_model_monitor():
    """مراقب نماذج وهمي للاختبار"""
    with patch('app.utils.tasks.model_monitor') as mock:
        mock.check_model_performance.return_value = True
        yield mock

@pytest.fixture
def mock_report_generator():
    """مولد تقارير وهمي للاختبار"""
    with patch('app.utils.tasks.report_generator') as mock:
        mock.generate_model_report.return_value = 'report1.pdf'
        mock.generate_monitoring_report.return_value = 'report2.pdf'
        yield mock

@pytest.mark.asyncio
async def test_update_models_task(mock_db, mock_model_updater):
    """اختبار مهمة تحديث النماذج"""
    result = await update_models_task()
    
    assert result['status'] == 'success'
    assert 'results' in result
    assert 'timestamp' in result
    
    mock_model_updater.check_and_update_models.assert_called_once()

@pytest.mark.asyncio
async def test_monitor_models_task(mock_db, mock_model_monitor):
    """اختبار مهمة مراقبة النماذج"""
    result = await monitor_models_task()
    
    assert result['status'] == 'success'
    assert 'results' in result
    assert len(result['results']) == 2
    
    assert mock_model_monitor.check_model_performance.call_count == 2

@pytest.mark.asyncio
async def test_generate_reports_task(mock_db, mock_report_generator):
    """اختبار مهمة توليد التقارير"""
    result = await generate_reports_task()
    
    assert result['status'] == 'success'
    assert 'results' in result
    assert len(result['results']) == 2
    
    assert mock_report_generator.generate_model_report.call_count == 2
    assert mock_report_generator.generate_monitoring_report.call_count == 2

@pytest.mark.asyncio
async def test_retrain_model_task(mock_model_updater):
    """اختبار مهمة إعادة تدريب النموذج"""
    new_data = {'feature1': [1, 2, 3], 'target': [0, 1, 0]}
    
    result = await retrain_model_task('test_model', new_data)
    
    assert result['status'] == 'success'
    assert result['model_id'] == 'test_model'
    assert 'timestamp' in result

def test_celery_config():
    """اختبار تكوين Celery"""
    assert celery_app.conf.task_serializer == 'json'
    assert celery_app.conf.result_serializer == 'json'
    assert celery_app.conf.accept_content == ['json']
    assert celery_app.conf.enable_utc is True

def test_celery_beat_schedule():
    """اختبار جدولة المهام الدورية"""
    schedule = celery_app.conf.beat_schedule
    
    assert 'update-models' in schedule
    assert 'monitor-models' in schedule
    assert 'generate-reports' in schedule
    
    assert schedule['update-models']['task'] == 'app.utils.tasks.update_models_task'
    assert schedule['monitor-models']['task'] == 'app.utils.tasks.monitor_models_task'
    assert schedule['generate-reports']['task'] == 'app.utils.tasks.generate_reports_task'

@pytest.mark.asyncio
async def test_task_error_handling(mock_db):
    """اختبار معالجة الأخطاء في المهام"""
    mock_db.query.side_effect = Exception("Database error")
    
    with pytest.raises(Exception):
        await update_models_task()
    
    with pytest.raises(Exception):
        await monitor_models_task()
    
    with pytest.raises(Exception):
        await generate_reports_task()

@pytest.mark.asyncio
async def test_task_retry(mock_model_updater):
    """اختبار إعادة محاولة المهام"""
    mock_model_updater.check_and_update_models.side_effect = Exception("Temporary error")
    
    with pytest.raises(Exception):
        await update_models_task()
    
    # التحقق من إعادة المحاولة
    assert update_models_task.retry.called
    assert update_models_task.retry.call_args[1]['countdown'] == 300  # 5 minutes 