import pytest
import numpy as np
from datetime import timedelta
import json
from app.utils.cache import cache_manager, NumpyEncoder
from unittest.mock import Mock, patch

@pytest.fixture
def sample_data():
    """بيانات للاختبار"""
    return {
        'array': np.array([1, 2, 3]),
        'number': 42,
        'string': 'test',
        'nested': {
            'array': np.array([4, 5, 6]),
            'float': np.float32(3.14)
        }
    }

def test_numpy_encoder():
    """اختبار مشفر NumPy"""
    encoder = NumpyEncoder()
    
    # اختبار مصفوفة NumPy
    assert encoder.default(np.array([1, 2, 3])) == [1, 2, 3]
    
    # اختبار عدد صحيح NumPy
    assert encoder.default(np.int32(42)) == 42
    
    # اختبار عدد عشري NumPy
    assert encoder.default(np.float32(3.14)) == 3.14
    
    # اختبار قيمة غير مدعومة
    with pytest.raises(TypeError):
        encoder.default(object())

def test_generate_cache_key(sample_data):
    """اختبار توليد مفتاح التخزين المؤقت"""
    key1 = cache_manager.generate_cache_key(sample_data)
    key2 = cache_manager.generate_cache_key(sample_data)
    
    assert isinstance(key1, str)
    assert key1 == key2  # نفس البيانات يجب أن تنتج نفس المفتاح
    
    # بيانات مختلفة يجب أن تنتج مفاتيح مختلفة
    different_data = sample_data.copy()
    different_data['number'] = 43
    different_key = cache_manager.generate_cache_key(different_data)
    assert key1 != different_key

@pytest.mark.asyncio
async def test_set_get(sample_data):
    """اختبار تخزين واسترجاع القيم"""
    key = 'test_key'
    
    # تخزين القيمة
    await cache_manager.set(key, sample_data)
    
    # استرجاع القيمة
    cached_data = await cache_manager.get(key)
    
    assert cached_data is not None
    assert cached_data['number'] == sample_data['number']
    assert cached_data['string'] == sample_data['string']
    assert isinstance(cached_data['array'], list)
    assert cached_data['array'] == sample_data['array'].tolist()

@pytest.mark.asyncio
async def test_set_with_ttl():
    """اختبار التخزين مع وقت انتهاء الصلاحية"""
    key = 'ttl_test'
    value = {'test': 'data'}
    ttl = timedelta(seconds=1)
    
    await cache_manager.set(key, value, ttl)
    
    # القيمة يجب أن تكون موجودة مباشرة
    assert await cache_manager.exists(key)
    
    # انتظار انتهاء الصلاحية
    import asyncio
    await asyncio.sleep(1.1)
    
    # القيمة يجب أن تكون قد انتهت صلاحيتها
    assert not await cache_manager.exists(key)

@pytest.mark.asyncio
async def test_delete():
    """اختبار حذف القيم"""
    key = 'delete_test'
    value = {'test': 'data'}
    
    await cache_manager.set(key, value)
    assert await cache_manager.exists(key)
    
    await cache_manager.delete(key)
    assert not await cache_manager.exists(key)

@pytest.mark.asyncio
async def test_clear():
    """اختبار مسح جميع القيم"""
    # تخزين بعض القيم
    await cache_manager.set('key1', 'value1')
    await cache_manager.set('key2', 'value2')
    
    # مسح جميع القيم
    await cache_manager.clear()
    
    # التحقق من عدم وجود القيم
    assert not await cache_manager.exists('key1')
    assert not await cache_manager.exists('key2')

@pytest.mark.asyncio
async def test_set_many_get_many():
    """اختبار تخزين واسترجاع عدة قيم"""
    data = {
        'key1': {'value': 1},
        'key2': {'value': 2},
        'key3': {'value': 3}
    }
    
    # تخزين القيم
    await cache_manager.set_many(data)
    
    # استرجاع القيم
    cached_data = await cache_manager.get_many(list(data.keys()))
    
    assert len(cached_data) == len(data)
    assert all(cached_data[key]['value'] == data[key]['value'] 
              for key in data.keys())

def test_redis_connection_error():
    """اختبار خطأ الاتصال بـ Redis"""
    with patch('redis.Redis') as mock_redis:
        mock_redis.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception):
            cache_manager.__init__() 