from typing import Dict, Any, Optional, Union
import redis.asyncio as redis
import json
from datetime import datetime, timedelta
import hashlib
import pickle
from app.core.config import settings
from app.utils.exceptions import CacheError
import logging
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """مشفر JSON مخصص للتعامل مع التواريخ"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class CustomJSONDecoder(json.JSONDecoder):
    """مفكك JSON مخصص للتعامل مع التواريخ"""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self, dct: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in dct.items():
            if isinstance(value, str):
                try:
                    dct[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return dct

class CacheManager:
    """مدير التخزين المؤقت"""
    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.use_local_cache = True
        self.max_local_cache_size = settings.cache.local_cache_size
        self.default_ttl = settings.cache.default_ttl
        self.key_prefix = settings.cache.key_prefix
        
    async def connect(self) -> None:
        """الاتصال بـ Redis"""
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis(
                    host=settings.redis.host,
                    port=settings.redis.port,
                    db=settings.redis.db,
                    password=settings.redis.password,
                    encoding=settings.redis.encoding,
                    decode_responses=True,
                    retry_on_timeout=True
                )
                logger.info("تم الاتصال بـ Redis بنجاح")
        except Exception as e:
            logger.error(f"فشل الاتصال بـ Redis: {str(e)}")
            self.use_local_cache = True
            
    async def test_connection(self) -> bool:
        """اختبار اتصال Redis"""
        try:
            if not self.redis_client:
                await self.connect()
            if self.redis_client:
                await self.redis_client.ping()
                return True
        except Exception as e:
            logger.error(f"فشل اختبار اتصال Redis: {str(e)}")
        return False
        
    async def get(self, key: str) -> Optional[Any]:
        """استرجاع قيمة من التخزين المؤقت"""
        try:
            # محاولة استرجاع من التخزين المحلي أولاً
            if self.use_local_cache and key in self.local_cache:
                cache_entry = self.local_cache[key]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['value']
                else:
                    del self.local_cache[key]
                    
            # محاولة استرجاع من Redis
            if await self.test_connection():
                full_key = f"{self.key_prefix}{key}"
                value = await self.redis_client.get(full_key)
                if value:
                    return json.loads(value, cls=CustomJSONDecoder)
                    
            return None
            
        except Exception as e:
            logger.error(f"خطأ في استرجاع القيمة: {str(e)}")
            return None
            
    async def set(self,
                  key: str,
                  value: Any,
                  ttl: Optional[int] = None) -> bool:
        """تخزين قيمة في التخزين المؤقت"""
        try:
            ttl = ttl or self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            # تخزين في التخزين المحلي
            if self.use_local_cache:
                self.local_cache[key] = {
                    'value': value,
                    'expiry': expiry
                }
                await self._cleanup_local_cache()
                
            # تخزين في Redis
            if await self.test_connection():
                full_key = f"{self.key_prefix}{key}"
                serialized_value = json.dumps(value, cls=CustomJSONEncoder)
                await self.redis_client.setex(full_key, ttl, serialized_value)
                
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تخزين القيمة: {str(e)}")
            raise CacheError(f"فشل تخزين القيمة: {str(e)}")
            
    async def delete(self, key: str) -> bool:
        """حذف قيمة من التخزين المؤقت"""
        try:
            # حذف من التخزين المحلي
            if key in self.local_cache:
                del self.local_cache[key]
                
            # حذف من Redis
            if await self.test_connection():
                full_key = f"{self.key_prefix}{key}"
                await self.redis_client.delete(full_key)
                
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حذف القيمة: {str(e)}")
            return False
            
    async def clear(self) -> bool:
        """مسح كل التخزين المؤقت"""
        try:
            # مسح التخزين المحلي
            self.local_cache.clear()
            
            # مسح Redis
            if await self.test_connection():
                await self.redis_client.flushdb()
                
            return True
            
        except Exception as e:
            logger.error(f"خطأ في مسح التخزين المؤقت: {str(e)}")
            return False
            
    async def generate_cache_key(self, data: Any) -> str:
        """توليد مفتاح للتخزين المؤقت"""
        try:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"خطأ في توليد مفتاح التخزين المؤقت: {str(e)}")
            return str(datetime.now().timestamp())
            
    async def _cleanup_local_cache(self) -> None:
        """تنظيف التخزين المحلي"""
        if len(self.local_cache) > self.max_local_cache_size:
            # حذف القيم منتهية الصلاحية
            current_time = datetime.now()
            expired_keys = [
                key for key, value in self.local_cache.items()
                if value['expiry'] < current_time
            ]
            for key in expired_keys:
                del self.local_cache[key]
                
            # حذف أقدم القيم إذا كان لا يزال كبيراً جداً
            if len(self.local_cache) > self.max_local_cache_size:
                sorted_items = sorted(
                    self.local_cache.items(),
                    key=lambda x: x[1]['expiry']
                )
                for key, _ in sorted_items[:len(sorted_items) - self.max_local_cache_size]:
                    del self.local_cache[key]

    async def close(self) -> None:
        """إغلاق اتصال Redis"""
        if self.redis_client:
            await self.redis_client.close()

cache_manager = CacheManager()

def cache_decorator(ttl: Optional[int] = None):
    """مزخرف للتخزين المؤقت"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # توليد مفتاح التخزين المؤقت
                cache_key = await cache_manager.generate_cache_key({
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
                
                # محاولة استرجاع من التخزين المؤقت
                cached_value = await cache_manager.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # تنفيذ الدالة وتخزين النتيجة
                result = await func(*args, **kwargs)
                await cache_manager.set(cache_key, result, ttl)
                return result
                
            except Exception as e:
                logger.error(f"خطأ في التخزين المؤقت: {str(e)}")
                return await func(*args, **kwargs)
                
        return wrapper
    return decorator
  