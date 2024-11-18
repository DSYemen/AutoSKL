import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.core.events import (
    create_start_app_handler,
    create_stop_app_handler,
    create_db_tables_handler
)
from fastapi import FastAPI

@pytest.fixture
def app():
    """تطبيق FastAPI للاختبار"""
    return FastAPI()

@pytest.fixture
def mock_db():
    """قاعدة بيانات وهمية للاختبار"""
    with patch('app.core.events.create_db_and_tables') as mock:
        yield mock

@pytest.fixture
def mock_model_updater():
    """محدث نماذج وهمي للاختبار"""
    with patch('app.core.events.start_model_update_scheduler') as start_mock, \
         patch('app.core.events.stop_model_update_scheduler') as stop_mock:
        yield {'start': start_mock, 'stop': stop_mock}

@pytest.fixture
def mock_cache():
    """ذاكرة مؤقتة وهمية للاختبار"""
    with patch('app.core.events.cache_manager') as mock:
        mock.redis_client.ping.return_value = True
        yield mock

@pytest.mark.asyncio
async def test_start_app_handler(app, mock_db, mock_model_updater, mock_cache):
    """اختبار معالج بدء التطبيق"""
    start_handler = create_start_app_handler(app)
    
    await start_handler()
    
    # التحقق من استدعاء الدوال المطلوبة
    mock_db.assert_called_once()
    mock_model_updater['start'].assert_called_once()
    mock_cache.redis_client.ping.assert_called_once()

@pytest.mark.asyncio
async def test_stop_app_handler(app, mock_model_updater, mock_cache):
    """اختبار معالج إيقاف التطبيق"""
    stop_handler = create_stop_app_handler(app)
    
    await stop_handler()
    
    # التحقق من استدعاء الدوال المطلوبة
    mock_model_updater['stop'].assert_called_once()
    mock_cache.clear.assert_called_once()
    mock_cache.redis_client.close.assert_called_once()

@pytest.mark.asyncio
async def test_db_tables_handler(mock_db):
    """اختبار معالج إنشاء جداول قاعدة البيانات"""
    handler = create_db_tables_handler()
    
    await handler()
    
    mock_db.assert_called_once()

@pytest.mark.asyncio
async def test_start_handler_redis_error(app, mock_db, mock_model_updater, mock_cache):
    """اختبار معالج البدء مع خطأ Redis"""
    mock_cache.redis_client.ping.return_value = False
    
    start_handler = create_start_app_handler(app)
    
    await start_handler()
    
    # يجب أن يستمر التطبيق حتى مع فشل Redis
    mock_db.assert_called_once()
    mock_model_updater['start'].assert_called_once()

@pytest.mark.asyncio
async def test_start_handler_db_error(app, mock_db, mock_model_updater, mock_cache):
    """اختبار معالج البدء مع خطأ قاعدة البيانات"""
    mock_db.side_effect = Exception("Database error")
    
    start_handler = create_start_app_handler(app)
    
    with pytest.raises(Exception):
        await start_handler()
    
    # يجب عدم بدء تحديث النماذج إذا فشلت قاعدة البيانات
    mock_model_updater['start'].assert_not_called()

@pytest.mark.asyncio
async def test_stop_handler_error_handling(app, mock_model_updater, mock_cache):
    """اختبار معالجة الأخطاء في معالج الإيقاف"""
    mock_cache.clear.side_effect = Exception("Cache error")
    
    stop_handler = create_stop_app_handler(app)
    
    with pytest.raises(Exception):
        await stop_handler()
    
    # يجب محاولة إيقاف كل شيء حتى مع وجود خطأ
    mock_model_updater['stop'].assert_called_once()
    mock_cache.redis_client.close.assert_called_once()

def test_event_handler_registration(app):
    """اختبار تسجيل معالجي الأحداث"""
    # تسجيل المعالجين
    app.add_event_handler("startup", create_start_app_handler(app))
    app.add_event_handler("shutdown", create_stop_app_handler(app))
    
    # التحقق من تسجيل المعالجين
    assert len(app.router.lifespan.startup_handlers) == 1
    assert len(app.router.lifespan.shutdown_handlers) == 1 