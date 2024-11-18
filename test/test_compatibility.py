import pytest
import sys
import platform
import pkg_resources
import importlib
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import create_app

client = TestClient(create_app())

def test_python_version():
    """اختبار توافق إصدار Python"""
    version = sys.version_info
    
    # التحقق من إصدار Python المدعوم
    assert version.major == 3
    assert version.minor >= 8, "يتطلب Python 3.8 أو أحدث"

def test_dependencies_versions():
    """اختبار إصدارات التبعيات"""
    requirements = pkg_resources.parse_requirements(
        Path('requirements.txt').read_text()
    )
    
    for req in requirements:
        pkg = pkg_resources.working_set.by_key.get(req.key)
        if pkg:
            assert pkg.version in req.specifier, \
                f"إصدار {req.key} غير متوافق: {pkg.version}"

def test_os_compatibility():
    """اختبار توافق نظام التشغيل"""
    system = platform.system().lower()
    
    # التحقق من دعم نظام التشغيل
    assert system in ['linux', 'darwin', 'windows'], \
        f"نظام التشغيل غير مدعوم: {system}"
    
    if system == 'windows':
        # اختبارات خاصة بـ Windows
        assert platform.release() >= '10', "يتطلب Windows 10 أو أحدث"

def test_database_compatibility():
    """اختبار توافق قاعدة البيانات"""
    from sqlalchemy import __version__ as sa_version
    
    # التحقق من توافق SQLAlchemy
    major, minor, *_ = sa_version.split('.')
    assert int(major) >= 1, "يتطلب SQLAlchemy 1.4 أو أحدث"
    
    if int(major) == 1:
        assert int(minor) >= 4

def test_ml_libraries_compatibility():
    """اختبار توافق مكتبات التعلم الآلي"""
    import sklearn
    import xgboost
    import lightgbm
    import catboost
    
    # التحقق من إصدارات المكتبات
    assert sklearn.__version__ >= '1.0'
    assert xgboost.__version__ >= '1.5'
    assert lightgbm.__version__ >= '3.3'
    assert catboost.__version__ >= '1.0'

def test_async_compatibility():
    """اختبار توافق البرمجة المتزامنة"""
    import asyncio
    
    # التحقق من دعم asyncio
    assert sys.version_info >= (3, 7), "يتطلب Python 3.7+ لدعم asyncio"
    
    # التحقق من توفر uvloop
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # uvloop اختياري

def test_frontend_compatibility():
    """اختبار توافق الواجهة الأمامية"""
    response = client.get("/")
    
    # التحقق من توافق HTML5
    assert "<!DOCTYPE html>" in response.text
    assert 'charset="UTF-8"' in response.text
    
    # التحقق من دعم CSS3
    assert "flex" in response.text
    assert "grid" in response.text

def test_api_versioning():
    """اختبار إصدارات API"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    # التحقق من وجود معلومات الإصدار
    assert "info" in schema
    assert "version" in schema["info"]
    
    version = schema["info"]["version"]
    assert version.count('.') >= 2  # تنسيق الإصدار الدلالي

def test_cache_compatibility():
    """اختبار توافق التخزين المؤقت"""
    import redis
    
    # التحقق من توافق Redis
    redis_version = redis.__version__
    assert redis_version >= '3.0.0'

def test_file_encoding():
    """اختبار ترميز الملفات"""
    source_files = Path('.').rglob('*.py')
    
    for file in source_files:
        try:
            content = file.read_text(encoding='utf-8')
            assert content.encode('utf-8').decode('utf-8') == content
        except UnicodeError:
            pytest.fail(f"خطأ في ترميز الملف: {file}")

def test_numpy_compatibility():
    """اختبار توافق NumPy"""
    import numpy as np
    
    # التحقق من توافق النوع
    arr = np.array([1, 2, 3])
    assert arr.dtype.kind in np.typecodes['AllInteger']
    
    # التحقق من تحويلات النوع
    float_arr = arr.astype(float)
    assert float_arr.dtype.kind in np.typecodes['AllFloat']

def test_pandas_compatibility():
    """اختبار توافق Pandas"""
    import pandas as pd
    
    # التحقق من معالجة التاريخ
    date_series = pd.date_range('2024-01-01', periods=5)
    assert date_series.dtype == 'datetime64[ns]'
    
    # التحقق من معالجة القيم المفقودة
    series = pd.Series([1, None, 3])
    assert pd.isna(series[1])

def test_json_compatibility():
    """اختبار توافق JSON"""
    import json
    import numpy as np
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    data = {
        'array': np.array([1, 2, 3]),
        'number': np.float32(1.5)
    }
    
    # التحقق من إمكانية تحويل البيانات إلى JSON
    json_str = json.dumps(data, cls=CustomEncoder)
    parsed_data = json.loads(json_str)
    assert isinstance(parsed_data['array'], list) 