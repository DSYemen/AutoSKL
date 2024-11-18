from typing import Dict, Any, List, Optional, Protocol, TypeVar, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from app.ml.data_processing import data_processor
from app.utils.cache import cache_manager
from app.utils.exceptions import PredictionError
import logging
from datetime import datetime
import joblib
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelType = TypeVar('ModelType', bound=BaseEstimator)
ArrayLike: TypeAlias = np.ndarray | pd.DataFrame | pd.Series
PredictionResult: TypeAlias = Dict[str, Any]

class PredictionService:
    """خدمة التنبؤ"""
    def __init__(self) -> None:
        self.model: Optional[ModelType] = None
        self.task_type: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.prediction_history: List[Dict[str, Any]] = []
        self.cache_enabled: bool = True
        self.batch_size: int = 1000
        self.predictions_dir = Path("predictions")
        self.predictions_dir.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def setup(self, 
                   model: ModelType,
                   task_type: str,
                   feature_names: List[str],
                   cache_enabled: bool = True) -> None:
        """إعداد خدمة التنبؤ مع خيارات إضافية"""
        try:
            self.model = model
            self.task_type = task_type
            self.feature_names = feature_names
            self.cache_enabled = cache_enabled
            logger.info(f"تم إعداد خدمة التنبؤ: {task_type}")
        except Exception as e:
            logger.error(f"خطأ في إعداد خدمة التنبؤ: {str(e)}")
            raise PredictionError(f"فشل إعداد خدمة التنبؤ: {str(e)}")
        
    def validate_input(self, 
                      data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
                      check_features: bool = True) -> pd.DataFrame:
        """تحقق محسن من صحة البيانات المدخلة"""
        try:
            # تحويل البيانات إلى DataFrame
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                raise PredictionError("نوع البيانات غير مدعوم")
                
            if check_features and self.feature_names:
                # التحقق من وجود الميزات المطلوبة
                missing_features = set(self.feature_names) - set(data.columns)
                if missing_features:
                    raise PredictionError(f"الميزات التالية مفقودة: {missing_features}")
                
                # التحقق من أنواع البيانات
                for col in data.columns:
                    if data[col].dtype not in ['int64', 'float64', 'object', 'bool']:
                        try:
                            data[col] = pd.to_numeric(data[col])
                        except:
                            logger.warning(f"تعذر تحويل العمود {col} إلى نوع رقمي")
                
            return data
            
        except Exception as e:
            raise PredictionError(f"خطأ في التحقق من صحة البيانات: {str(e)}")
            
    def format_prediction(self, 
                         predictions: np.ndarray,
                         probabilities: Optional[np.ndarray] = None,
                         include_metadata: bool = True) -> List[PredictionResult]:
        """تنسيق محسن للتنبؤات"""
        try:
            results = []
            
            for i, pred in enumerate(predictions):
                result = {
                    'prediction': pred.item() if isinstance(pred, np.generic) else pred,
                    'timestamp': datetime.now().isoformat()
                }
                
                if probabilities is not None:
                    if probabilities.ndim == 2:
                        result['probabilities'] = {
                            f'class_{j}': prob.item()
                            for j, prob in enumerate(probabilities[i])
                        }
                    else:
                        result['probability'] = probabilities[i].item()
                        
                if include_metadata:
                    result['metadata'] = {
                        'model_type': self.task_type,
                        'version': getattr(self.model, 'version', 'unknown')
                    }
                    
                results.append(result)
                
            return results
            
        except Exception as e:
            raise PredictionError(f"خطأ في تنسيق التنبؤات: {str(e)}")
            
    async def predict(self, 
                     data: Union[pd.DataFrame, Dict[str, Any]],
                     return_probabilities: bool = False,
                     include_metadata: bool = True) -> List[PredictionResult]:
        """إجراء تنبؤات مع تحسينات إضافية"""
        try:
            if self.model is None:
                raise PredictionError("لم يتم تحميل النموذج")
                
            # التحقق من صحة البيانات
            df = self.validate_input(data)
            
            # التحقق من التخزين المؤقت
            if self.cache_enabled:
                cache_key = await cache_manager.generate_cache_key(df.values)
                cached_predictions = await cache_manager.get(cache_key)
                if cached_predictions is not None:
                    logger.info("تم استرجاع التنبؤات من الذاكرة المؤقتة")
                    return cached_predictions
                
            # معالجة البيانات
            X = await asyncio.to_thread(data_processor.transform_new_data, df)
            
            # إجراء التنبؤات
            predictions = await asyncio.to_thread(self.model.predict, X)
            probabilities = None
            
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = await asyncio.to_thread(self.model.predict_proba, X)
                except Exception as e:
                    logger.warning(f"فشل حساب الاحتمالات: {str(e)}")
                    
            # تنسيق النتائج
            results = self.format_prediction(predictions, probabilities, include_metadata)
            
            # حفظ التنبؤات في التاريخ
            await self._save_prediction_history(results)
            
            # تخزين النتائج مؤقتاً
            if self.cache_enabled:
                await cache_manager.set(cache_key, results)
                
            return results
            
        except Exception as e:
            logger.error(f"خطأ في إجراء التنبؤات: {str(e)}")
            raise PredictionError(f"خطأ في إجراء التنبؤات: {str(e)}")
            
    async def predict_batch(self,
                          data: List[Union[pd.DataFrame, Dict[str, Any]]],
                          batch_size: Optional[int] = None,
                          return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """إجراء تنبؤات للدفعات مع تحسينات"""
        try:
            batch_size = batch_size or self.batch_size
            all_results = []
            
            # تقسيم البيانات إلى دفعات
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                logger.info(f"معالجة الدفعة {i//batch_size + 1}")
                
                if isinstance(batch[0], dict):
                    batch_df = pd.DataFrame(batch)
                else:
                    batch_df = pd.concat(batch, ignore_index=True)
                    
                batch_results = await self.predict(batch_df, return_probabilities)
                all_results.extend(batch_results)
                
            return all_results
            
        except Exception as e:
            raise PredictionError(f"خطأ في إجراء تنبؤات الدفعات: {str(e)}")
            
    async def _save_prediction_history(self, predictions: List[Dict[str, Any]]) -> None:
        """حفظ تاريخ التنبؤات"""
        try:
            self.prediction_history.extend(predictions)
            
            # حفظ التنبؤات في ملف
            history_file = self.predictions_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.joblib"
            await asyncio.to_thread(joblib.dump, predictions, history_file)
            
        except Exception as e:
            logger.warning(f"فشل حفظ تاريخ التنبؤات: {str(e)}")
            
    async def get_prediction_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبؤات"""
        try:
            if not self.prediction_history:
                return {}
                
            predictions_df = pd.DataFrame(self.prediction_history)
            
            return {
                'total_predictions': len(self.prediction_history),
                'unique_predictions': len(predictions_df['prediction'].unique()),
                'last_prediction_time': predictions_df['timestamp'].max(),
                'prediction_distribution': predictions_df['prediction'].value_counts().to_dict(),
                'performance_metrics': await self._calculate_performance_metrics(predictions_df)
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب إحصائيات التنبؤات: {str(e)}")
            return {}
            
    async def _calculate_performance_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """حساب مقاييس الأداء للتنبؤات"""
        try:
            metrics = {
                'average_response_time': predictions_df.get('response_time', pd.Series()).mean(),
                'success_rate': (predictions_df['status'] == 'success').mean() if 'status' in predictions_df else 1.0,
                'error_rate': (predictions_df['status'] == 'error').mean() if 'status' in predictions_df else 0.0
            }
            
            if 'confidence' in predictions_df:
                metrics['average_confidence'] = predictions_df['confidence'].mean()
                metrics['confidence_distribution'] = predictions_df['confidence'].describe().to_dict()
                
            return metrics
            
        except Exception as e:
            logger.warning(f"فشل حساب مقاييس الأداء: {str(e)}")
            return {}

prediction_service = PredictionService() 