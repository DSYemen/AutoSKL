import pytest
import logging
from pathlib import Path
from unittest.mock import patch, Mock
from app.core.logging_config import setup_logging, get_logger

@pytest.fixture
def mock_logger():
    """مسجل وهمي للاختبار"""
    with patch('logging.getLogger') as mock:
        mock.return_value = Mock(spec=logging.Logger)
        yield mock

@pytest.fixture
def temp_log_dir(tmp_path):
    """مجلد سجلات مؤقت للاختبار"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir

def test_setup_logging(mock_logger, temp_log_dir):
    """اختبار إعداد التسجيل"""
    with patch('app.core.logging_config.Path', return_value=temp_log_dir):
        setup_logging()
        
        # التحقق من إعداد المسجل الرئيسي
        mock_logger.assert_called()
        root_logger = mock_logger.return_value
        root_logger.setLevel.assert_called_once()
        
        # التحقق من إضافة المعالجات
        assert root_logger.addHandler.call_count == 2  # ملف ووحدة تحكم

def test_get_logger():
    """اختبار الحصول على مسجل"""
    logger = get_logger('test_component')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'app.test_component'

def test_log_file_creation(temp_log_dir):
    """اختبار إنشاء ملف السجل"""
    with patch('app.core.logging_config.Path', return_value=temp_log_dir):
        setup_logging()
        log_file = temp_log_dir / 'app.log'
        assert log_file.exists()

def test_log_rotation(temp_log_dir):
    """اختبار تدوير ملفات السجل"""
    with patch('app.core.logging_config.Path', return_value=temp_log_dir):
        setup_logging()
        logger = logging.getLogger()
        
        # كتابة بيانات كبيرة للسجل
        large_data = 'x' * 1024 * 1024  # 1MB
        for _ in range(15):  # لتجاوز حد 10MB
            logger.info(large_data)
            
        # التحقق من وجود ملفات السجل المدورة
        log_files = list(temp_log_dir.glob('app.log*'))
        assert len(log_files) > 1

def test_component_loggers():
    """اختبار مسجلات المكونات"""
    loggers = {
        'data_processing': get_logger('ml.data_processing'),
        'model_selection': get_logger('ml.model_selection'),
        'api': get_logger('api')
    }
    
    for logger in loggers.values():
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

def test_log_formatting():
    """اختبار تنسيق السجلات"""
    with patch('logging.Handler.format') as mock_format:
        logger = get_logger('test')
        logger.info('test message')
        
        # التحقق من تنسيق الرسالة
        log_record = mock_format.call_args[0][0]
        assert log_record.name == 'app.test'
        assert log_record.levelname == 'INFO'
        assert log_record.msg == 'test message'

def test_external_loggers_level():
    """اختبار مستوى مسجلات المكتبات الخارجية"""
    external_loggers = {
        'uvicorn.access': logging.WARNING,
        'sqlalchemy.engine': logging.WARNING
    }
    
    for logger_name, expected_level in external_loggers.items():
        logger = logging.getLogger(logger_name)
        assert logger.level == expected_level

def test_log_directory_creation():
    """اختبار إنشاء مجلد السجلات"""
    log_dir = Path('logs')
    if log_dir.exists():
        for file in log_dir.glob('*'):
            file.unlink()
        log_dir.rmdir()
        
    setup_logging()
    assert log_dir.exists()
    assert log_dir.is_dir()

def test_error_logging():
    """اختبار تسجيل الأخطاء"""
    logger = get_logger('test')
    with patch.object(logger, 'error') as mock_error:
        try:
            raise ValueError('test error')
        except Exception as e:
            logger.error('An error occurred: %s', str(e))
            
        mock_error.assert_called_once_with('An error occurred: %s', 'test error') 