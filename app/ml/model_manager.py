from typing import Dict, Any, Optional, List, Tuple, Protocol
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.base import BaseEstimator
import sklearn
import shutil
import hashlib
from app.core.config import settings
from app.utils.exceptions import ModelNotFoundError
import logging
from typing_extensions import TypeAlias
import asyncio
import platform
from prometheus_client import Counter, Gauge, Histogram
import aiofiles
import os
from app.schemas.model import ModelStatus
import psutil

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelInfo: TypeAlias = Dict[str, Any]
ModelMetadata: TypeAlias = Dict[str, Any]

# Prometheus متريكس
MODEL_SAVE_DURATION = Histogram('model_save_duration_seconds', 'Time taken to save model')
MODEL_LOAD_DURATION = Histogram('model_load_duration_seconds', 'Time taken to load model')
MODELS_COUNT = Gauge('models_total', 'Total number of models')
MODEL_SIZE = Gauge('model_size_bytes', 'Model size in bytes', ['model_id'])
MODEL_OPERATIONS = Counter('model_operations_total', 'Model operations', ['operation'])
MODEL_ERRORS = Counter('model_errors_total', 'Model operation errors', ['error_type'])

class NumpyJSONEncoder(json.JSONEncoder):
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
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

class ModelProtocol(Protocol):
    """بروتوكول للنماذج"""
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

class ModelManager:
    """مدير النماذج"""
    def __init__(self) -> None:
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.current_models: Dict[str, BaseEstimator] = {}
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.backup_dir = self.models_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.max_cache_size = settings.cache.local_cache_size
        self.cache_ttl = settings.cache.local_cache_ttl
        self.max_backups = 5  # عدد النسخ الاحتياطية المحتفظ بها لكل نموذج
        
    async def save_model(self, 
                        model: ModelProtocol, 
                        model_id: str, 
                        metadata: ModelMetadata,
                        create_backup: bool = True) -> None:
        """حفظ النموذج وبياناته الوصفية مع تحسينات"""
        try:
            with MODEL_SAVE_DURATION.time():
                # إنشاء نسخة احتياطية إذا كان النموذج موجوداً
                if create_backup and (self.models_dir / f"{model_id}.joblib").exists():
                    await self._create_backup(model_id)
                
                # إضافة معلومات إضافية للبيانات الوصفية
                metadata.update({
                    'last_updated': datetime.utcnow().isoformat(),
                    'model_hash': await self._calculate_model_hash(model),
                    'framework_version': settings.app.version,
                    'scikit_learn_version': sklearn.__version__,
                    'python_version': platform.python_version(),
                    'system_info': {
                        'platform': platform.platform(),
                        'processor': platform.processor(),
                        'python_implementation': platform.python_implementation(),
                        'memory_info': dict(psutil.virtual_memory()._asdict())
                    },
                    'status': ModelStatus.ACTIVE.value
                })
                
                # حفظ النموذج والبيانات الوصفية
                model_path = self.models_dir / f"{model_id}.joblib"
                metadata_path = self.models_dir / f"{model_id}_metadata.json"
                
                await asyncio.to_thread(joblib.dump, model, model_path)
                async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(metadata, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))
                
                # تحديث المقاييس
                model_size = model_path.stat().st_size
                MODEL_SIZE.labels(model_id=model_id).set(model_size)
                MODEL_OPERATIONS.labels(operation='save').inc()
                MODELS_COUNT.inc()
                
                # تحديث الذاكرة المؤقتة
                self.current_models[model_id] = model
                self.model_cache[model_id] = {
                    'model': model,
                    'metadata': metadata,
                    'loaded_at': datetime.utcnow(),
                    'size': model_size,
                    'access_count': 0
                }
                
                # تنظيف الذاكرة المؤقتة إذا تجاوزت الحد الأقصى
                await self._cleanup_cache()
                
                logger.info(f"تم حفظ النموذج {model_id} بنجاح")
                
        except Exception as e:
            logger.error(f"خطأ في حفظ النموذج {model_id}: {str(e)}")
            MODEL_OPERATIONS.labels(operation='save_error').inc()
            MODEL_ERRORS.labels(error_type='save').inc()
            raise
            
    async def load_model(self, 
                        model_id: str, 
                        use_cache: bool = True) -> Tuple[ModelProtocol, ModelMetadata]:
        """تحميل النموذج وبياناته الوصفية مع دعم التخزين المؤقت"""
        try:
            with MODEL_LOAD_DURATION.time():
                # التحقق من الذاكرة المؤقتة
                if use_cache and model_id in self.model_cache:
                    cache_entry = self.model_cache[model_id]
                    cache_age = (datetime.utcnow() - cache_entry['loaded_at']).total_seconds()
                    
                    if cache_age < self.cache_ttl:
                        logger.info(f"تم استرجاع النموذج {model_id} من الذاكرة المؤقتة")
                        MODEL_OPERATIONS.labels(operation='cache_hit').inc()
                        return cache_entry['model'], cache_entry['metadata']
                
                model_path = self.models_dir / f"{model_id}.joblib"
                metadata_path = self.models_dir / f"{model_id}_metadata.json"
                
                if not model_path.exists():
                    raise ModelNotFoundError(f"النموذج {model_id} غير موجود")
                    
                # تحميل النموذج والبيانات الوصفية
                model = await asyncio.to_thread(joblib.load, model_path)
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
                
                # التحقق من سلامة النموذج
                current_hash = await self._calculate_model_hash(model)
                if current_hash != metadata.get('model_hash'):
                    logger.warning(f"تحذير: تغير hash النموذج {model_id}")
                    MODEL_OPERATIONS.labels(operation='hash_mismatch').inc()
                
                # تحديث الذاكرة المؤقتة
                self.current_models[model_id] = model
                self.model_cache[model_id] = {
                    'model': model,
                    'metadata': metadata,
                    'loaded_at': datetime.utcnow(),
                    'size': model_path.stat().st_size
                }
                
                MODEL_OPERATIONS.labels(operation='load').inc()
                return model, metadata
                
        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج {model_id}: {str(e)}")
            MODEL_OPERATIONS.labels(operation='load_error').inc()
            raise
            
    async def delete_model(self, model_id: str, create_backup: bool = True) -> None:
        """حذف النموذج مع خيار النسخ الاحتياطي"""
        try:
            if create_backup:
                await self._create_backup(model_id)
            
            model_path = self.models_dir / f"{model_id}.joblib"
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            # حذف الملفات
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
            # تنظيف الذاكرة المؤقتة
            self.current_models.pop(model_id, None)
            self.model_cache.pop(model_id, None)
            
            # تحديث المقاييس
            MODEL_SIZE.labels(model_id=model_id).set(0)
            MODEL_OPERATIONS.labels(operation='delete').inc()
            MODELS_COUNT.dec()
            
            logger.info(f"تم حذف النموذج {model_id} بنجاح")
            
        except Exception as e:
            logger.error(f"خطأ في حذف النموذج {model_id}: {str(e)}")
            MODEL_OPERATIONS.labels(operation='delete_error').inc()
            raise
            
    async def _create_backup(self, model_id: str) -> None:
        """إنشاء نسخة احتياطية من النموذج"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"{model_id}_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # نسخ ملفات النموذج
            await asyncio.to_thread(shutil.copy2,
                self.models_dir / f"{model_id}.joblib",
                backup_path / f"{model_id}.joblib"
            )
            await asyncio.to_thread(shutil.copy2,
                self.models_dir / f"{model_id}_metadata.json",
                backup_path / f"{model_id}_metadata.json"
            )
            
            MODEL_OPERATIONS.labels(operation='backup').inc()
            logger.info(f"تم إنشاء نسخة احتياطية للنموذج {model_id}")
            
        except Exception as e:
            logger.warning(f"فشل إنشاء نسخة احتياطية للنموذج {model_id}: {str(e)}")
            MODEL_OPERATIONS.labels(operation='backup_error').inc()
            
    async def _calculate_model_hash(self, model: ModelProtocol) -> str:
        """حساب hash للنموذج للتحقق من سلامته"""
        try:
            model_bytes = await asyncio.to_thread(joblib.dumps, model)
            return hashlib.sha256(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"فشل حساب hash النموذج: {str(e)}")
            return ""
            
    async def _cleanup_cache(self) -> None:
        """تنظيف الذاكرة المؤقتة"""
        if len(self.model_cache) > self.max_cache_size:
            # حذف أقدم النماذج من الذاكرة المؤقتة
            sorted_cache = sorted(
                self.model_cache.items(),
                key=lambda x: x[1]['loaded_at']
            )
            for model_id, _ in sorted_cache[:len(sorted_cache) - self.max_cache_size]:
                self.model_cache.pop(model_id)
                logger.debug(f"تم إزالة النموذج {model_id} من الذاكرة المؤقتة")

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """الحصول على معلومات النموذج"""
        try:
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            if not metadata_path.exists():
                return None
                
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
                
            return metadata
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على معلومات النموذج: {str(e)}")
            return None
            
    async def list_models(self) -> List[str]:
        """الحصول على قائمة النماذج المتوفرة"""
        try:
            model_files = list(self.models_dir.glob("*.joblib"))
            return [f.stem for f in model_files]
        except Exception as e:
            logger.error(f"خطأ في الحصول على قائمة النماذج: {str(e)}")
            return []
            
    async def get_model_size(self, model_id: str) -> int:
        """الحصول على حجم النموذج بالبايت"""
        try:
            model_path = self.models_dir / f"{model_id}.joblib"
            return model_path.stat().st_size if model_path.exists() else 0
        except Exception as e:
            logger.error(f"خطأ في الحصول على حجم النموذج: {str(e)}")
            return 0

    async def get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على مقاييس النموذج"""
        try:
            model_path = self.models_dir / f"{model_id}.joblib"
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                return None
                
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
                
            metrics = {
                'model_size': model_path.stat().st_size,
                'last_updated': metadata.get('last_updated'),
                'evaluation_results': metadata.get('evaluation_results', {}),
                'feature_importance': metadata.get('feature_importance', {})
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على مقاييس النموذج: {str(e)}")
            return None
            
    async def update_model_metadata(self,
                                  model_id: str,
                                  metadata_updates: Dict[str, Any]) -> bool:
        """تحديث البيانات الوصفية للنموذج"""
        try:
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if not metadata_path.exists():
                return False
                
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
                
            # تحديث البيانات الوصفية
            metadata.update(metadata_updates)
            metadata['last_updated'] = datetime.utcnow().isoformat()
            
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2))
                
            MODEL_OPERATIONS.labels(operation='metadata_update').inc()
            return True
            
        except Exception as e:
            logger.error(f"خطأ في تحديث البيانات الوصفية: {str(e)}")
            MODEL_OPERATIONS.labels(operation='metadata_update_error').inc()
            return False
            
    async def cleanup_old_backups(self, model_id: str) -> None:
        """تنظيف النسخ الاحتياطية القديمة"""
        try:
            backup_pattern = f"{model_id}_*"
            backups = sorted(
                self.backup_dir.glob(backup_pattern),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # الاحتفاظ بآخر N نسخ احتياطية فقط
            if len(backups) > self.max_backups:
                for backup in backups[self.max_backups:]:
                    if backup.is_dir():
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()
                        
                logger.info(f"تم تنظيف النسخ الاحتياطية القديمة للنموذج {model_id}")
                
        except Exception as e:
            logger.error(f"خطأ في تنظيف النسخ الاحتياطية: {str(e)}")
            
    async def get_model_dependencies(self, model_id: str) -> Dict[str, Any]:
        """الحصول على تبعيات النموذج"""
        try:
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if not metadata_path.exists():
                return {}
                
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
                
            return {
                'scikit_learn_version': metadata.get('scikit_learn_version'),
                'python_version': metadata.get('python_version'),
                'framework_version': metadata.get('framework_version'),
                'system_info': metadata.get('system_info', {})
            }
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على تبعيات النموذج: {str(e)}")
            return {}
            
    async def get_model_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """الحصول على تاريخ أداء النموذج"""
        try:
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if not metadata_path.exists():
                return []
                
            async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.loads(await f.read())
                
            return metadata.get('performance_history', [])
            
        except Exception as e:
            logger.error(f"خطأ في الحصول على تاريخ الأداء: {str(e)}")
            return []
            
    async def export_model(self, 
                          model_id: str,
                          export_format: str = 'joblib',
                          include_metadata: bool = True) -> Optional[str]:
        """تصدير النموذج بتنسيق محدد"""
        try:
            model_path = self.models_dir / f"{model_id}.joblib"
            if not model_path.exists():
                return None
                
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = export_dir / f"{model_id}_{timestamp}.{export_format}"
            
            if export_format == 'joblib':
                await asyncio.to_thread(shutil.copy2, model_path, export_path)
                
                if include_metadata:
                    metadata_path = self.models_dir / f"{model_id}_metadata.json"
                    if metadata_path.exists():
                        await asyncio.to_thread(
                            shutil.copy2,
                            metadata_path,
                            export_path.with_suffix('.json')
                        )
                        
            else:
                raise ValueError(f"تنسيق التصدير غير مدعوم: {export_format}")
                
            MODEL_OPERATIONS.labels(operation='export').inc()
            return str(export_path)
            
        except Exception as e:
            logger.error(f"خطأ في تصدير النموذج: {str(e)}")
            MODEL_OPERATIONS.labels(operation='export_error').inc()
            return None
            
    async def import_model(self,
                          import_path: str,
                          new_model_id: Optional[str] = None) -> Optional[str]:
        """استيراد نموذج"""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                raise FileNotFoundError(f"الملف غير موجود: {import_path}")
                
            if new_model_id is None:
                new_model_id = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            model_path = self.models_dir / f"{new_model_id}.joblib"
            
            # نسخ ملف النموذج
            await asyncio.to_thread(shutil.copy2, import_path, model_path)
            
            # نسخ البيانات الوصفية إذا كانت موجودة
            metadata_path = import_path.with_suffix('.json')
            if metadata_path.exists():
                await asyncio.to_thread(
                    shutil.copy2,
                    metadata_path,
                    self.models_dir / f"{new_model_id}_metadata.json"
                )
                
            MODEL_OPERATIONS.labels(operation='import').inc()
            return new_model_id
            
        except Exception as e:
            logger.error(f"خطأ في استيراد النموذج: {str(e)}")
            MODEL_OPERATIONS.labels(operation='import_error').inc()
            return None

model_manager = ModelManager() 