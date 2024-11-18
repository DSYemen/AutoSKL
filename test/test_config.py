import pytest
from pathlib import Path
import yaml
from app.core.config import settings
from pydantic import ValidationError

@pytest.fixture
def sample_config():
    """تكوين نموذجي للاختبار"""
    return {
        'app': {
            'name': 'AutoML Framework',
            'version': '1.0.0',
            'debug': True,
            'host': '0.0.0.0',
            'port': 8000
        },
        'database': {
            'url': 'sqlite:///./test.db'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        },
        'ml': {
            'task_types': [
                'regression',
                'classification',
                'clustering',
                'dimensionality_reduction'
            ],
            'model_selection': {
                'cv_folds': 5,
                'n_trials': 100,
                'timeout': 3600
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42
            },
            'monitoring': {
                'drift_threshold': 0.05,
                'performance_threshold': 0.95,
                'check_interval': 3600
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/app.log'
        }
    }

def test_app_settings(sample_config):
    """اختبار إعدادات التطبيق"""
    assert settings.app.name == sample_config['app']['name']
    assert settings.app.version == sample_config['app']['version']
    assert settings.app.debug == sample_config['app']['debug']
    assert settings.app.host == sample_config['app']['host']
    assert settings.app.port == sample_config['app']['port']

def test_database_settings(sample_config):
    """اختبار إعدادات قاعدة البيانات"""
    assert settings.database.url == sample_config['database']['url']

def test_redis_settings(sample_config):
    """اختبار إعدادات Redis"""
    assert settings.redis.host == sample_config['redis']['host']
    assert settings.redis.port == sample_config['redis']['port']

def test_ml_settings(sample_config):
    """اختبار إعدادات التعلم الآلي"""
    assert all(task in settings.ml.task_types 
              for task in sample_config['ml']['task_types'])
    assert settings.ml.model_selection.cv_folds == sample_config['ml']['model_selection']['cv_folds']
    assert settings.ml.training.test_size == sample_config['ml']['training']['test_size']
    assert settings.ml.monitoring.drift_threshold == sample_config['ml']['monitoring']['drift_threshold']

def test_logging_settings(sample_config):
    """اختبار إعدادات التسجيل"""
    assert settings.logging.level == sample_config['logging']['level']
    assert settings.logging.format == sample_config['logging']['format']
    assert settings.logging.file == sample_config['logging']['file']

def test_invalid_config():
    """اختبار تكوين غير صالح"""
    invalid_config = {
        'app': {
            'port': 'invalid_port'  # يجب أن يكون رقماً
        }
    }
    
    with pytest.raises(ValidationError):
        settings.parse_obj(invalid_config)

def test_missing_required_fields():
    """اختبار حقول مطلوبة مفقودة"""
    incomplete_config = {
        'app': {
            'name': 'Test App'
            # نقص حقول مطلوبة
        }
    }
    
    with pytest.raises(ValidationError):
        settings.parse_obj(incomplete_config)

def test_config_file_exists():
    """اختبار وجود ملف التكوين"""
    config_path = Path('config.yaml')
    assert config_path.exists()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert 'app' in config
        assert 'database' in config
        assert 'ml' in config

def test_environment_override():
    """اختبار تجاوز البيئة"""
    import os
    
    # تعيين متغير بيئة
    os.environ['APP_PORT'] = '9000'
    
    # إعادة تحميل الإعدادات
    from importlib import reload
    from app.core import config
    reload(config)
    
    # التحقق من تجاوز القيمة
    assert config.settings.app.port == 9000
    
    # تنظيف
    del os.environ['APP_PORT']

def test_nested_settings():
    """اختبار الإعدادات المتداخلة"""
    assert hasattr(settings.ml.model_selection, 'cv_folds')
    assert hasattr(settings.ml.training, 'test_size')
    assert hasattr(settings.ml.monitoring, 'drift_threshold')

def test_settings_immutability():
    """اختبار عدم قابلية التغيير"""
    with pytest.raises(Exception):
        settings.app.port = 9000 