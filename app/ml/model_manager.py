from typing import Dict, Any, Optional, List, Tuple, Protocol
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.base import BaseEstimator
import shutil
import hashlib
from app.core.config import settings
from app.utils.exceptions import ModelNotFoundError
import logging
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelInfo: TypeAlias = Dict[str, Any]
ModelMetadata: TypeAlias = Dict[str, Any]

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
        
    async def save_model(self, 
                        model: ModelProtocol, 
                        model_id: str, 
                        metadata: ModelMetadata,
                        create_backup: bool = True) -> None:
        """حفظ النموذج وبياناته الوصفية مع تحسينات"""
        try:
            # إنشاء نسخة احتياطية إذا كان النموذج موجوداً
            if create_backup and (self.models_dir / f"{model_id}.joblib").exists():
                await self._create_backup(model_id)
            
            # إضافة معلومات إضافية للبيانات الوصفية
            metadata.update({
                'last_updated': datetime.utcnow().isoformat(),
                'model_hash': await self._calculate_model_hash(model),
                'framework_version': settings.app.version,
                'scikit_learn_version': sklearn.__version__
            })
            
            # حفظ النموذج والبيانات الوصفية
            model_path = self.models_dir / f"{model_id}.joblib"
            metadata_path = self.models_dir / f"{model_id}_metadata.joblib"
            
            await asyncio.to_thread(joblib.dump, model, model_path)
            await asyncio.to_thread(joblib.dump, metadata, metadata_path)
            
            # تحديث الذاكرة المؤقتة
            self.current_models[model_id] = model
            self.model_cache[model_id] = {
                'model': model,
                'metadata': metadata,
                'loaded_at': datetime.utcnow()
            }
            
            logger.info(f"تم حفظ النموذج {model_id} بنجاح")
            
        except Exception as e:
            logger.error(f"خطأ في حفظ النموذج {model_id}: {str(e)}")
            raise
            
    async def load_model(self, model_id: str, use_cache: bool = True) -> Tuple[ModelProtocol, ModelMetadata]:
        """تحميل النموذج وبياناته الوصفية مع دعم التخزين المؤقت"""
        try:
            # التحقق من الذاكرة المؤقتة
            if use_cache and model_id in self.model_cache:
                cache_entry = self.model_cache[model_id]
                cache_age = (datetime.utcnow() - cache_entry['loaded_at']).total_seconds()
                
                if cache_age < settings.ml.model_management.model_cache_ttl:
                    logger.info(f"تم استرجاع النموذج {model_id} من الذاكرة المؤقتة")
                    return cache_entry['model'], cache_entry['metadata']
            
            model_path = self.models_dir / f"{model_id}.joblib"
            metadata_path = self.models_dir / f"{model_id}_metadata.joblib"
            
            if not model_path.exists():
                raise ModelNotFoundError(f"النموذج {model_id} غير موجود")
                
            # تحميل النموذج والبيانات الوصفية
            model = await asyncio.to_thread(joblib.load, model_path)
            metadata = await asyncio.to_thread(joblib.load, metadata_path)
            
            # التحقق من سلامة النموذج
            current_hash = await self._calculate_model_hash(model)
            if current_hash != metadata.get('model_hash'):
                logger.warning(f"تحذير: تغير hash النموذج {model_id}")
            
            # تحديث الذاكرة المؤقتة
            self.current_models[model_id] = model
            self.model_cache[model_id] = {
                'model': model,
                'metadata': metadata,
                'loaded_at': datetime.utcnow()
            }
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج {model_id}: {str(e)}")
            raise
            
    async def delete_model(self, model_id: str, create_backup: bool = True) -> None:
        """حذف النموذج مع خيار النسخ الاحتياطي"""
        try:
            model_path = self.models_dir / f"{model_id}.joblib"
            metadata_path = self.models_dir / f"{model_id}_metadata.joblib"
            
            if create_backup:
                await self._create_backup(model_id)
            
            # حذف الملفات
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
            # تنظيف الذاكرة المؤقتة
            self.current_models.pop(model_id, None)
            self.model_cache.pop(model_id, None)
            
            logger.info(f"تم حذف النموذج {model_id} بنجاح")
            
        except Exception as e:
            logger.error(f"خطأ في حذف النموذج {model_id}: {str(e)}")
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
                self.models_dir / f"{model_id}_metadata.joblib",
                backup_path / f"{model_id}_metadata.joblib"
            )
            
            logger.info(f"تم إنشاء نسخة احتياطية للنموذج {model_id}")
            
        except Exception as e:
            logger.warning(f"فشل إنشاء نسخة احتياطية للنموذج {model_id}: {str(e)}")
            
    async def _calculate_model_hash(self, model: ModelProtocol) -> str:
        """حساب hash للنموذج للتحقق من سلامته"""
        try:
            model_bytes = await asyncio.to_thread(joblib.dumps, model)
            return hashlib.sha256(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"فشل حساب hash النموذج: {str(e)}")
            return ""

    # ... باقي الدوال بنفس النمط من التحديثات

model_manager = ModelManager() 