from typing import Dict, Any, Optional, List, Protocol
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import Model, ModelUpdate
from app.ml.model_evaluation import model_evaluator
from app.ml.model_selection import model_selector
from app.ml.data_processing import data_processor
from app.utils.alerts import alert_manager
from app.core.config import settings
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelData: TypeAlias = Dict[str, Any]
UpdateResult: TypeAlias = Dict[str, Any]

class ModelUpdater:
    """محدث النماذج"""
    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.update_history: List[Dict[str, Any]] = []
        self.improvement_threshold = settings.ml.training.improvement_threshold
        
    async def update_model_if_needed(self,
                                   model_id: str,
                                   new_data: pd.DataFrame,
                                   session: AsyncSession) -> UpdateResult:
        """تحديث النموذج إذا كان هناك تحسن في الأداء"""
        try:
            # تحميل النموذج الحالي ومعلوماته
            model_info = await self._get_model_info(model_id, session)
            if not model_info:
                raise ValueError(f"النموذج {model_id} غير موجود")
                
            current_model = model_info['model']
            metadata = model_info['metadata']
            
            # تقييم النموذج الحالي على البيانات الجديدة
            X_new, y_new = await data_processor.process_data(
                new_data,
                metadata['target_column'],
                metadata['task_type']
            )
            
            current_performance = await model_evaluator.evaluate_model(
                current_model,
                X_new,
                y_new,
                metadata['task_type'],
                metadata['feature_names']
            )
            
            # تدريب نموذج جديد على البيانات الجديدة
            new_model, new_params = await model_selector.select_best_model(
                X_new,
                y_new,
                metadata['task_type']
            )
            
            new_performance = await model_evaluator.evaluate_model(
                new_model,
                X_new,
                y_new,
                metadata['task_type'],
                metadata['feature_names']
            )
            
            # مقارنة الأداء
            improvement = self._calculate_improvement(
                current_performance['metrics'],
                new_performance['metrics'],
                metadata['task_type']
            )
            
            update_info = {
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'current_performance': current_performance,
                'new_performance': new_performance,
                'improvement': improvement,
                'update_needed': improvement > self.improvement_threshold
            }
            
            if update_info['update_needed']:
                # تحديث النموذج
                await self._update_model(
                    model_id,
                    new_model,
                    new_params,
                    new_performance,
                    current_performance,
                    session
                )
                
                # إرسال تنبيه
                await alert_manager.send_model_update_alert(
                    model_id,
                    current_performance['metrics'],
                    new_performance['metrics'],
                    settings.alerts.recipients
                )
                
            # تحديث التاريخ
            self.update_history.append(update_info)
            
            return update_info
            
        except Exception as e:
            logger.error(f"خطأ في تحديث النموذج {model_id}: {str(e)}")
            raise
            
    async def _get_model_info(self,
                             model_id: str,
                             session: AsyncSession) -> Optional[ModelData]:
        """استرجاع معلومات النموذج"""
        try:
            stmt = select(Model).where(Model.model_id == model_id)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                return None
                
            return {
                'model': model,
                'metadata': model.model_metadata
            }
            
        except Exception as e:
            logger.error(f"خطأ في استرجاع معلومات النموذج: {str(e)}")
            return None
            
    def _calculate_improvement(self,
                             current_metrics: Dict[str, float],
                             new_metrics: Dict[str, float],
                             task_type: str) -> float:
        """حساب نسبة التحسن في الأداء"""
        try:
            if task_type == 'classification':
                metric_key = 'accuracy'
            else:
                metric_key = 'r2'
                
            current_value = current_metrics.get(metric_key, 0)
            new_value = new_metrics.get(metric_key, 0)
            
            if current_value == 0:
                return float('inf') if new_value > 0 else 0
                
            return (new_value - current_value) / current_value
            
        except Exception as e:
            logger.error(f"خطأ في حساب نسبة التحسن: {str(e)}")
            return 0
            
    async def _update_model(self,
                          model_id: str,
                          new_model: Any,
                          new_params: Dict[str, Any],
                          new_performance: Dict[str, Any],
                          current_performance: Dict[str, Any],
                          session: AsyncSession) -> None:
        """تحديث النموذج في قاعدة البيانات"""
        try:
            # إنشاء سجل التحديث
            update = ModelUpdate(
                model_id=model_id,
                update_type='retrain',
                performance_before=current_performance['metrics'],
                performance_after=new_performance['metrics'],
                update_metadata={
                    'parameters': new_params,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            session.add(update)
            
            # تحديث النموذج
            stmt = select(Model).where(Model.model_id == model_id)
            result = await session.execute(stmt)
            model = result.scalar_one()
            
            model.model_metadata.update({
                'last_updated': datetime.utcnow().isoformat(),
                'current_performance': new_performance,
                'update_history': self.update_history
            })
            
            await session.commit()
            logger.info(f"تم تحديث النموذج {model_id} بنجاح")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"خطأ في تحديث النموذج في قاعدة البيانات: {str(e)}")
            raise
            
    def start_scheduler(self) -> None:
        """بدء جدولة تحديثات النموذج"""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("تم بدء جدولة تحديثات النموذج")
        except Exception as e:
            logger.error(f"خطأ في بدء الجدولة: {str(e)}")
            
    def stop_scheduler(self) -> None:
        """إيقاف جدولة تحديثات النموذج"""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown()
                logger.info("تم إيقاف جدولة تحديثات النموذج")
        except Exception as e:
            logger.error(f"خطأ في إيقاف الجدولة: {str(e)}")
            
    def get_update_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التحديث"""
        try:
            if not self.update_history:
                return {}
                
            updates_df = pd.DataFrame(self.update_history)
            return {
                'total_updates': len(self.update_history),
                'successful_updates': len(updates_df[updates_df['update_needed']]),
                'average_improvement': updates_df['improvement'].mean(),
                'last_update': updates_df['timestamp'].max(),
                'update_frequency': self._calculate_update_frequency(updates_df)
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب إحصائيات التحديث: {str(e)}")
            return {}
            
    def _calculate_update_frequency(self, updates_df: pd.DataFrame) -> str:
        """حساب معدل تكرار التحديثات"""
        try:
            if len(updates_df) < 2:
                return "غير متوفر"
                
            timestamps = pd.to_datetime(updates_df['timestamp'])
            intervals = timestamps.diff().dropna()
            avg_interval = intervals.mean()
            
            if avg_interval < timedelta(hours=1):
                return f"كل {int(avg_interval.total_seconds() / 60)} دقيقة"
            elif avg_interval < timedelta(days=1):
                return f"كل {int(avg_interval.total_seconds() / 3600)} ساعة"
            else:
                return f"كل {int(avg_interval.days)} يوم"
                
        except Exception as e:
            logger.error(f"خطأ في حساب معدل تكرار التحديثات: {str(e)}")
            return "غير متوفر"

model_updater = ModelUpdater()

# تصدير الدوال المساعدة
async def start_model_update_scheduler() -> None:
    model_updater.start_scheduler()

async def stop_model_update_scheduler() -> None:
    model_updater.stop_scheduler() 