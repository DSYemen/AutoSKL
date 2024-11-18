from typing import Any, Optional, Dict, List, Union, TypeVar, Protocol
from redis.asyncio import Redis
import json
import hashlib
import numpy as np
import zlib
from datetime import timedelta, datetime
from app.core.config import settings
from app.utils.exceptions import CacheError
import logging
import asyncio
from functools import wraps
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
T = TypeVar('T')
CacheKey: TypeAlias = str
CacheValue: TypeAlias = Union[str, bytes, Dict[str, Any], List[Any]]

class CacheBackend(Protocol):
    """بروتوكول لخلفيات التخزين المؤقت"""
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: bytes, ex: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...

class NumpyEncoder(json.JSONEncoder):
    """مشفر JSON مخصص للتعامل مع مصفوفات NumPy"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class CacheManager:
    """مدير التخزين المؤقت"""
    def __init__(self) -> None:
        self.redis_client: Optional[Redis] = None
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = timedelta(seconds=settings.cache.default_ttl)
        self.compression_threshold = 1024  # حد الضغط بالبايت
        self.key_prefix = settings.cache.key_prefix
        
        try:
            self.redis_client = Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=settings.redis.decode_responses,
                encoding=settings.redis.encoding,
                retry_on_timeout=settings.redis.retry_on_timeout,
                health_check_interval=settings.redis.health_check_interval
            )
            logger.info("تم الاتصال بـ Redis بنجاح")
        except Exception as e:
            logger.warning(f"فشل الاتصال بـ Redis، سيتم استخدام التخزين المؤقت المحلي: {str(e)}")
            
    async def test_connection(self) -> bool:
        """اختبار اتصال Redis"""
        try:
            if self.redis_client:
                return await self.redis_client.ping()
            return False
        except Exception as e:
            logger.error(f"فشل اختبار اتصال Redis: {str(e)}")
            return False

    async def close(self) -> None:
        """إغلاق اتصال Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("تم إغلاق اتصال Redis")
        except Exception as e:
            logger.error(f"خطأ في إغلاق اتصال Redis: {str(e)}")

    async def set(self,
                  key: CacheKey,
                  value: CacheValue,
                  ttl: Optional[timedelta] = None,
                  compress: bool = True) -> None:
        """تخزين قيمة في الذاكرة المؤقتة"""
        try:
            key = self._format_key(key)
            serialized_value = self._serialize_value(value)
            
            if compress and len(serialized_value) > self.compression_threshold:
                serialized_value = self._compress_data(serialized_value)
                key = f"compressed:{key}"
            
            if self.redis_client and await self.test_connection():
                await self.redis_client.set(
                    key,
                    serialized_value,
                    ex=(ttl or self.default_ttl).total_seconds()
                )
            else:
                self.local_cache[key] = {
                    'value': serialized_value,
                    'expires_at': datetime.utcnow() + (ttl or self.default_ttl)
                }
                
            logger.debug(f"تم تخزين القيمة في المفتاح: {key}")
            
        except Exception as e:
            logger.error(f"خطأ في تخزين القيمة: {str(e)}")
            raise CacheError(f"فشل تخزين القيمة: {str(e)}")
            
    async def get(self, key: CacheKey) -> Optional[T]:
        """استرجاع قيمة من الذاكرة المؤقتة"""
        try:
            key = self._format_key(key)
            is_compressed = key.startswith("compressed:")
            
            if self.redis_client and await self.test_connection():
                value = await self.redis_client.get(key)
            else:
                cache_item = self.local_cache.get(key)
                if cache_item and datetime.utcnow() < cache_item['expires_at']:
                    value = cache_item['value']
                else:
                    value = None
                    
            if value:
                if is_compressed:
                    value = self._decompress_data(value)
                return self._deserialize_value(value)
                
            return None
            
        except Exception as e:
            logger.error(f"خطأ في استرجاع القيمة: {str(e)}")
            return None

    def _format_key(self, key: str) -> str:
        """تنسيق مفتاح التخزين المؤقت"""
        return f"{self.key_prefix}{key}"
        
    def _serialize_value(self, value: Any) -> str:
        """تحويل القيمة إلى سلسلة نصية"""
        return json.dumps(value, cls=NumpyEncoder)
        
    def _deserialize_value(self, value: str) -> Any:
        """تحويل السلسلة النصية إلى قيمة"""
        return json.loads(value)
        
    def _compress_data(self, data: str) -> bytes:
        """ضغط البيانات"""
        return zlib.compress(data.encode())
        
    def _decompress_data(self, data: bytes) -> str:
        """فك ضغط البيانات"""
        return zlib.decompress(data).decode()
        
    def generate_cache_key(self, data: Any) -> str:
        """توليد مفتاح التخزين المؤقت"""
        try:
            if isinstance(data, np.ndarray):
                data = data.tobytes()
            elif not isinstance(data, (str, bytes)):
                data = json.dumps(data, sort_keys=True, cls=NumpyEncoder).encode()
                
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            logger.error(f"خطأ في توليد مفتاح التخزين المؤقت: {str(e)}")
            raise CacheError(f"فشل توليد مفتاح التخزين المؤقت: {str(e)}")
            
    async def delete(self, key: str) -> None:
        """حذف قيمة من الذاكرة المؤقتة"""
        try:
            key = self._format_key(key)
            if self.redis_client and self.redis_client.ping():
                await self.redis_client.delete(key)
            else:
                self.local_cache.pop(key, None)
                
            logger.debug(f"تم حذف المفتاح: {key}")
            
        except Exception as e:
            logger.error(f"خطأ في حذف القيمة: {str(e)}")
            raise CacheError(f"فشل حذف القيمة: {str(e)}")
            
    async def clear(self) -> None:
        """مسح جميع القيم من الذاكرة المؤقتة"""
        try:
            if self.redis_client and self.redis_client.ping():
                await self.redis_client.flushdb()
            else:
                self.local_cache.clear()
                
            logger.info("تم مسح الذاكرة المؤقتة")
            
        except Exception as e:
            logger.error(f"خطأ في مسح الذاكرة المؤقتة: {str(e)}")
            raise CacheError(f"فشل مسح الذاكرة المؤقتة: {str(e)}")
            
    async def exists(self, key: str) -> bool:
        """التحقق من وجود مفتاح في الذاكرة المؤقتة"""
        try:
            key = self._format_key(key)
            if self.redis_client and self.redis_client.ping():
                return bool(await self.redis_client.exists(key))
            else:
                return key in self.local_cache
                
        except Exception as e:
            logger.error(f"خطأ في التحقق من وجود المفتاح: {str(e)}")
            return False

cache_manager = CacheManager()

def cache_decorator(ttl: Optional[timedelta] = None):
    """مزخرف للتخزين المؤقت"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # توليد مفتاح فريد
            cache_key = f"{func.__name__}:{hashlib.sha256(str(args).encode()).hexdigest()}"
            
            # محاولة استرجاع القيمة من الذاكرة المؤقتة
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # تنفيذ الدالة وتخزين النتيجة
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
  