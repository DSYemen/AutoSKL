from typing import Dict, Any, Optional, Union
import redis.asyncio as redis
import json
from json.decoder import JSONDecodeError
from datetime import datetime, timedelta
import hashlib
import pickle
from app.core.config import settings
from app.utils.exceptions import CacheError
import logging
import asyncio
from functools import wraps
import threading
from collections import OrderedDict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """مشفر JSON مخصص للتعامل مع الأنواع المختلفة"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class LocalCache:
    """تخزين مؤقت محلي باستخدام OrderedDict"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """استرجاع قيمة من التخزين المؤقت المحلي"""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if expiry > datetime.now():
                    # تحريك العنصر إلى نهاية القائمة (LRU)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # حذف العنصر منتهي الصلاحية
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        """تخزين قيمة في التخزين المؤقت المحلي"""
        with self.lock:
            # تنظيف الذاكرة المؤقتة إذا تجاوزت الحد الأقصى
            if len(self.cache) >= self.max_size:
                # حذف أقدم عنصر (LRU)
                self.cache.popitem(last=False)
            
            expiry = datetime.now() + timedelta(seconds=ttl)
            self.cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """حذف قيمة من التخزين المؤقت المحلي"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self) -> None:
        """مسح كل التخزين المؤقت المحلي"""
        with self.lock:
            self.cache.clear()

    def cleanup(self) -> None:
        """تنظيف العناصر منتهية الصلاحية"""
        with self.lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if expiry <= now
            ]
            for key in expired_keys:
                del self.cache[key]

class CacheManager:
    """مدير التخزين المؤقت"""
    def __init__(self) -> None:
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache = LocalCache(max_size=settings.cache.local_cache_size)
        self.use_redis = True
        self.default_ttl = settings.cache.default_ttl
        self.key_prefix = settings.cache.key_prefix
        self.json_encoder = CustomJSONEncoder

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
            logger.warning(f"فشل الاتصال بـ Redis: {str(e)}. سيتم استخدام التخزين المؤقت المحلي.")
            self.use_redis = False

    async def test_connection(self) -> bool:
        """اختبار اتصال Redis"""
        try:
            if not self.redis_client:
                await self.connect()
            if self.redis_client and self.use_redis:
                await self.redis_client.ping()
                return True
        except Exception as e:
            logger.warning(f"فشل اختبار اتصال Redis: {str(e)}. سيتم استخدام التخزين المؤقت المحلي.")
            self.use_redis = False
        return False

    async def get(self, key: str) -> Optional[Any]:
        """استرجاع قيمة من التخزين المؤقت"""
        try:
            # محاولة استرجاع من التخزين المحلي أولاً
            value = self.local_cache.get(key)
            if value is not None:
                return value

            # محاولة استرجاع من Redis إذا كان متاحاً
            if self.use_redis and await self.test_connection():
                full_key = f"{self.key_prefix}{key}"
                value = await self.redis_client.get(full_key)
                if value:
                    try:
                        decoded_value = json.loads(value)
                        # تخزين في التخزين المحلي
                        self.local_cache.set(key, decoded_value, self.default_ttl)
                        return decoded_value
                    except JSONDecodeError:
                        return value

            return None

        except Exception as e:
            logger.error(f"خطأ في استرجاع القيمة: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """تخزين قيمة في التخزين المؤقت"""
        try:
            ttl = ttl or self.default_ttl

            # تخزين في التخزين المحلي
            self.local_cache.set(key, value, ttl)

            # تخزين في Redis إذا كان متاحاً
            if self.use_redis and await self.test_connection():
                full_key = f"{self.key_prefix}{key}"
                try:
                    # استخدام المشفر المخصص
                    serialized_value = json.dumps(value, cls=self.json_encoder)
                    await self.redis_client.setex(full_key, ttl, serialized_value)
                except Exception as e:
                    logger.warning(f"فشل تحويل القيمة إلى JSON: {str(e)}")
                    # محاولة تخزين القيمة كنص
                    try:
                        if isinstance(value, BaseModel):
                            string_value = json.dumps(value.model_dump(), cls=self.json_encoder)
                        else:
                            string_value = str(value)
                        await self.redis_client.setex(full_key, ttl, string_value)
                    except Exception as e2:
                        logger.error(f"فشل تخزين القيمة كنص: {str(e2)}")
                        return False

            return True

        except Exception as e:
            logger.error(f"خطأ في تخزين القيمة: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """حذف قيمة من التخزين المؤقت"""
        try:
            # حذف من التخزين المحلي
            self.local_cache.delete(key)

            # حذف من Redis إذا كان متاحاً
            if self.use_redis and await self.test_connection():
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

            # مسح Redis إذا كان متاحاً
            if self.use_redis and await self.test_connection():
                await self.redis_client.flushdb()

            return True

        except Exception as e:
            logger.error(f"خطأ في مسح التخزين المؤقت: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """تنظيف التخزين المؤقت"""
        try:
            # تنظيف التخزين المحلي
            self.local_cache.cleanup()

        except Exception as e:
            logger.error(f"خطأ في تنظيف التخزين المؤقت: {str(e)}")

    async def close(self) -> None:
        """إغلاق اتصالات التخزين المؤقت"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("تم إغلاق اتصال Redis")
        except Exception as e:
            logger.error(f"خطأ في إغلاق اتصال Redis: {str(e)}")

cache_manager = CacheManager()

def cache_decorator(ttl: Optional[int] = None):
    """مزخرف للتخزين المؤقت"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # توليد مفتاح التخزين المؤقت
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
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
  