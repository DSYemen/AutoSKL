from typing import Dict, Any, List, Optional, Protocol, TypeVar
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from app.db.models import ModelMetrics, DataDrift
from app.utils.alerts import alert_manager
from app.core.config import settings
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import asyncio
from prometheus_client import Counter, Gauge, Histogram
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
MetricsData: TypeAlias = Dict[str, Any]
DriftResult: TypeAlias = Dict[str, Any]

# Prometheus متريكس
DRIFT_DETECTED = Counter('model_drift_detected_total', 'Number of drift detections', ['model_id', 'feature'])
PERFORMANCE_SCORE = Gauge('model_performance_score', 'Current model performance score', ['model_id', 'metric'])
MONITORING_DURATION = Histogram('model_monitoring_duration_seconds', 'Time spent monitoring model')

class ModelMonitor:
    """مراقب النماذج"""
    def __init__(self) -> None:
        self.drift_threshold = settings.ml.monitoring.drift_threshold
        self.performance_threshold = settings.ml.monitoring.performance_threshold
        self.scaler = StandardScaler()
        self.monitoring_history: List[Dict[str, Any]] = []
        
    async def start_monitoring(self,
                             model_id: str,
                             reference_data: pd.DataFrame,
                             session: AsyncSession) -> None:
        """بدء مراقبة النموذج"""
        try:
            logger.info(f"بدء مراقبة النموذج {model_id}")
            
            while True:
                with MONITORING_DURATION.time():
                    # جمع البيانات الحالية
                    current_data = await self._collect_current_data(model_id)
                    
                    if len(current_data) >= settings.ml.monitoring.min_samples_drift:
                        # اكتشاف انحراف البيانات
                        drift_results = await self.detect_data_drift(
                            model_id,
                            reference_data,
                            current_data,
                            session
                        )
                        
                        # تحليل الأداء
                        performance_results = await self.check_model_performance(
                            model_id,
                            await self._get_current_metrics(model_id, session),
                            session
                        )
                        
                        # تحديث المقاييس
                        await self._update_metrics(model_id, drift_results, performance_results)
                        
                        # حفظ نتائج المراقبة
                        await self._save_monitoring_results(
                            model_id,
                            drift_results,
                            performance_results,
                            session
                        )
                        
                await asyncio.sleep(settings.ml.monitoring.check_interval)
                
        except Exception as e:
            logger.error(f"خطأ في مراقبة النموذج {model_id}: {str(e)}")
            raise
            
    async def check_model_performance(self,
                                    model_id: str,
                                    current_metrics: Dict[str, float],
                                    session: AsyncSession) -> Dict[str, Any]:
        """التحقق من أداء النموذج مع تحليل متقدم"""
        try:
            # استرجاع المقاييس السابقة
            historical_metrics = await self._get_historical_metrics(model_id, session)
            
            if not historical_metrics:
                return {'status': 'no_history', 'metrics': current_metrics}
                
            # تحليل الاتجاه
            trend_analysis = await self._analyze_performance_trend(historical_metrics)
            
            # حساب الإحصائيات
            performance_stats = self._calculate_performance_statistics(historical_metrics)
            
            # التحقق من الأداء الحالي
            current_performance = current_metrics.get('accuracy', current_metrics.get('r2', 0))
            performance_threshold = performance_stats['mean'] * self.performance_threshold
            
            is_degraded = current_performance < performance_threshold
            
            if is_degraded:
                await alert_manager.send_model_performance_alert(
                    model_id,
                    {
                        'current': current_performance,
                        'threshold': performance_threshold,
                        'trend': trend_analysis
                    },
                    self.performance_threshold,
                    settings.alerts.recipients
                )
                
            # تحديث مقياس Prometheus
            PERFORMANCE_SCORE.labels(model_id=model_id, metric='overall').set(current_performance)
                
            return {
                'status': 'degraded' if is_degraded else 'stable',
                'current_metrics': current_metrics,
                'historical_stats': performance_stats,
                'trend_analysis': trend_analysis
            }
            
        except Exception as e:
            logger.error(f"خطأ في التحقق من أداء النموذج: {str(e)}")
            raise
            
    async def detect_data_drift(self,
                              model_id: str,
                              reference_data: pd.DataFrame,
                              current_data: pd.DataFrame,
                              db: AsyncSession) -> Dict[str, Any]:
        """اكتشاف انحراف البيانات مع تحليلات متقدمة"""
        try:
            drift_results = {}
            feature_importance = {}
            
            for column in reference_data.columns:
                # حساب مقاييس الانحراف المتعددة
                drift_metrics = self._calculate_drift_metrics(
                    reference_data[column],
                    current_data[column]
                )
                
                # حساب أهمية الميزة
                feature_importance[column] = self._calculate_feature_importance(
                    reference_data[column],
                    current_data[column]
                )
                
                drift_detected = any(metric['is_drift'] for metric in drift_metrics.values())
                
                if drift_detected:
                    DRIFT_DETECTED.labels(model_id=model_id, feature=column).inc()
                
                drift_results[column] = {
                    'metrics': drift_metrics,
                    'importance': feature_importance[column],
                    'drift_detected': drift_detected
                }
                
                # حفظ نتائج الانحراف
                await self._save_drift_results(
                    model_id, column, drift_detected,
                    drift_metrics['ks_test']['statistic'] if 'ks_test' in drift_metrics else 0,
                    db
                )
                
            # تحليل الانحراف الكلي
            overall_drift = self._analyze_overall_drift(drift_results)
            
            if overall_drift['is_significant']:
                await alert_manager.send_data_drift_alert(
                    model_id,
                    {
                        'drift_results': drift_results,
                        'overall_analysis': overall_drift
                    },
                    settings.alerts.recipients
                )
                
            return {
                'feature_drift': drift_results,
                'overall_analysis': overall_drift,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"خطأ في اكتشاف انحراف البيانات: {str(e)}")
            raise
            
    def _calculate_drift_metrics(self,
                               reference: pd.Series,
                               current: pd.Series) -> Dict[str, Any]:
        """حساب مقاييس الانحراف المتعددة"""
        metrics = {}
        
        # اختبار Kolmogorov-Smirnov
        if reference.dtype in ['int64', 'float64']:
            statistic, p_value = stats.ks_2samp(reference, current)
            metrics['ks_test'] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_drift': p_value < self.drift_threshold
            }
            
            # اختبار Jensen-Shannon divergence
            try:
                js_divergence = self._calculate_js_divergence(reference, current)
                metrics['js_divergence'] = {
                    'value': float(js_divergence),
                    'is_drift': js_divergence > 0.1
                }
            except:
                pass
                
        else:
            # اختبار Chi-square للبيانات الفئوية
            chi2, p_value = self._calculate_chi_square(reference, current)
            metrics['chi_square'] = {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'is_drift': p_value < self.drift_threshold
            }
            
        return metrics
        
    def _calculate_feature_importance(self,
                                    reference: pd.Series,
                                    current: pd.Series) -> float:
        """حساب أهمية الميزة في اكتشاف الانحراف"""
        try:
            if reference.dtype in ['int64', 'float64']:
                # استخدام mutual information للبيانات العددية
                reference_scaled = self.scaler.fit_transform(reference.values.reshape(-1, 1))
                current_scaled = self.scaler.transform(current.values.reshape(-1, 1))
                return float(mutual_info_score(reference_scaled.ravel(), current_scaled.ravel()))
            else:
                # استخدام تغير التوزيع للبيانات الفئوية
                ref_dist = reference.value_counts(normalize=True)
                curr_dist = current.value_counts(normalize=True)
                return float(np.abs(ref_dist - curr_dist).mean())
        except:
            return 0.0
            
    def _analyze_overall_drift(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل الانحراف الكلي"""
        drifted_features = [
            feature for feature, result in drift_results.items()
            if result['drift_detected']
        ]
        
        total_features = len(drift_results)
        drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0
        
        return {
            'is_significant': drift_ratio > 0.3,  # انحراف كبير إذا تأثر أكثر من 30% من الميزات
            'drift_ratio': drift_ratio,
            'drifted_features': drifted_features,
            'severity': 'high' if drift_ratio > 0.5 else 'medium' if drift_ratio > 0.3 else 'low'
        }
        
    async def _get_historical_metrics(self,
                                    model_id: str,
                                    db: AsyncSession,
                                    days: int = 30) -> List[Dict[str, Any]]:
        """استرجاع المقاييس التاريخية"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(ModelMetrics).where(
                ModelMetrics.model_id == model_id,
                ModelMetrics.timestamp >= cutoff_date
            ).order_by(ModelMetrics.timestamp)
            
            result = await db.execute(stmt)
            metrics = result.scalars().all()
            
            return [{'metrics': m.metrics, 'timestamp': m.timestamp} for m in metrics]
            
        except Exception as e:
            logger.error(f"خطأ في استرجاع المقاييس التاريخية: {str(e)}")
            return []
            
    async def _analyze_performance_trend(self,
                                 historical_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تحليل اتجاه الأداء"""
        try:
            performance_values = [
                m['metrics'].get('accuracy', m['metrics'].get('r2', 0))
                for m in historical_metrics
            ]
            
            # حساب معدل التغير
            changes = np.diff(performance_values)
            trend_direction = 'improving' if np.mean(changes) > 0 else 'degrading'
            
            # حساب التقلب
            volatility = np.std(changes) if len(changes) > 0 else 0
            
            return {
                'direction': trend_direction,
                'volatility': float(volatility),
                'stability': 'stable' if volatility < 0.1 else 'unstable'
            }
            
        except Exception as e:
            logger.error(f"خطأ في تحليل اتجاه الأداء: {str(e)}")
            return {}
            
    async def _save_monitoring_results(self,
                                     model_id: str,
                                     drift_results: Dict[str, Any],
                                     performance_results: Dict[str, Any],
                                     session: AsyncSession) -> None:
        """حفظ نتائج المراقبة"""
        try:
            monitoring_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'model_id': model_id,
                'drift_results': drift_results,
                'performance_results': performance_results
            }
            
            self.monitoring_history.append(monitoring_record)
            
            # الاحتفاظ بآخر 1000 سجل فقط
            if len(self.monitoring_history) > 1000:
                self.monitoring_history = self.monitoring_history[-1000:]
                
        except Exception as e:
            logger.error(f"خطأ في حفظ نتائج المراقبة: {str(e)}")
            
    async def get_monitoring_summary(self,
                                   model_id: str,
                                   days: int = 30) -> Dict[str, Any]:
        """الحصول على ملخص المراقبة"""
        try:
            recent_history = [
                record for record in self.monitoring_history
                if record['model_id'] == model_id and
                datetime.fromisoformat(record['timestamp']) > datetime.utcnow() - timedelta(days=days)
            ]
            
            if not recent_history:
                return {}
                
            return {
                'total_checks': len(recent_history),
                'drift_frequency': sum(
                    1 for record in recent_history
                    if record['drift_results']['overall_analysis']['is_significant']
                ) / len(recent_history),
                'performance_degradation_frequency': sum(
                    1 for record in recent_history
                    if record['performance_results']['status'] == 'degraded'
                ) / len(recent_history),
                'last_check': recent_history[-1]['timestamp'],
                'most_drifted_features': self._get_most_drifted_features(recent_history)
            }
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء ملخص المراقبة: {str(e)}")
            return {}
            
    def _get_most_drifted_features(self,
                                  history: List[Dict[str, Any]],
                                  top_n: int = 5) -> List[Dict[str, Any]]:
        """الحصول على الميزات الأكثر انحرافاً"""
        try:
            feature_drift_count = {}
            
            for record in history:
                for feature, result in record['drift_results']['feature_drift'].items():
                    if result['drift_detected']:
                        feature_drift_count[feature] = feature_drift_count.get(feature, 0) + 1
                        
            return sorted(
                [
                    {
                        'feature': feature,
                        'drift_frequency': count / len(history)
                    }
                    for feature, count in feature_drift_count.items()
                ],
                key=lambda x: x['drift_frequency'],
                reverse=True
            )[:top_n]
            
        except Exception as e:
            logger.error(f"خطأ في حساب الميزات الأكثر انحرافاً: {str(e)}")
            return []

model_monitor = ModelMonitor() 