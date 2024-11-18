from typing import Dict, Any, List, Optional, Protocol, TypeVar, Union, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
import shap
from app.utils.exceptions import ModelEvaluationError
import logging
from datetime import datetime
import joblib
from pathlib import Path
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelType = TypeVar('ModelType')
ArrayLike: TypeAlias = np.ndarray | pd.DataFrame | pd.Series
MetricFunction: TypeAlias = Callable[[ArrayLike, ArrayLike], float]

class ModelEvaluator:
    """مقيم النماذج"""
    def __init__(self) -> None:
        self.task_type: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Optional[Dict[str, float]] = None
        self.shap_values: Optional[np.ndarray] = None
        self.evaluation_history: List[Dict[str, Any]] = []
        self.evaluation_dir = Path("evaluations")
        self.evaluation_dir.mkdir(exist_ok=True)
        
    async def evaluate_model(self,
                           model: ModelType,
                           X: ArrayLike,
                           y: ArrayLike,
                           task_type: str,
                           feature_names: List[str],
                           sample_weight: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """تقييم شامل للنموذج"""
        try:
            self.task_type = task_type
            evaluation_start = datetime.now()
            
            if task_type == 'classification':
                metrics = await self._evaluate_classification(model, X, y, sample_weight)
            elif task_type == 'regression':
                metrics = await self._evaluate_regression(model, X, y, sample_weight)
            elif task_type == 'clustering':
                metrics = await self._evaluate_clustering(model, X)
            else:
                raise ModelEvaluationError(f"نوع المهمة غير مدعوم: {task_type}")
                
            # حساب أهمية الميزات وقيم SHAP
            feature_importance = await self._calculate_feature_importance(model, X, feature_names)
            shap_analysis = await self._calculate_shap_values(model, X, feature_names)
            
            # تجميع نتائج التقييم
            evaluation_results = {
                'metrics': metrics,
                'feature_importance': feature_importance,
                'shap_analysis': shap_analysis,
                'evaluation_time': (datetime.now() - evaluation_start).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            # حفظ نتائج التقييم
            await self._save_evaluation_results(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"خطأ في تقييم النموذج: {str(e)}")
            raise ModelEvaluationError(f"فشل تقييم النموذج: {str(e)}")
            
    async def _evaluate_classification(self,
                                    model: ModelType,
                                    X: ArrayLike,
                                    y: ArrayLike,
                                    sample_weight: Optional[ArrayLike] = None) -> Dict[str, Any]:
        """تقييم نموذج التصنيف مع مقاييس متقدمة"""
        try:
            y_pred = model.predict(X)
            metrics = {
                'accuracy': accuracy_score(y, y_pred, sample_weight=sample_weight),
                'precision_macro': precision_score(y, y_pred, average='macro', sample_weight=sample_weight),
                'recall_macro': recall_score(y, y_pred, average='macro', sample_weight=sample_weight),
                'f1_macro': f1_score(y, y_pred, average='macro', sample_weight=sample_weight),
                'precision_weighted': precision_score(y, y_pred, average='weighted', sample_weight=sample_weight),
                'recall_weighted': recall_score(y, y_pred, average='weighted', sample_weight=sample_weight),
                'f1_weighted': f1_score(y, y_pred, average='weighted', sample_weight=sample_weight)
            }
            
            # إضافة مقاييس متقدمة إذا كانت الاحتمالات متوفرة
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)
                
                # ROC AUC للتصنيف الثنائي والمتعدد
                try:
                    if y_prob.shape[1] == 2:  # تصنيف ثنائي
                        metrics['roc_auc'] = roc_auc_score(y, y_prob[:, 1], sample_weight=sample_weight)
                        metrics['average_precision'] = average_precision_score(y, y_prob[:, 1], sample_weight=sample_weight)
                        
                        # منحنيات ROC و PR
                        fpr, tpr, _ = roc_curve(y, y_prob[:, 1], sample_weight=sample_weight)
                        precision, recall, _ = precision_recall_curve(y, y_prob[:, 1], sample_weight=sample_weight)
                        
                        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                        metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
                    else:  # تصنيف متعدد
                        metrics['roc_auc_ovr'] = roc_auc_score(y, y_prob, multi_class='ovr', sample_weight=sample_weight)
                        metrics['roc_auc_ovo'] = roc_auc_score(y, y_prob, multi_class='ovo', sample_weight=sample_weight)
                except Exception as e:
                    logger.warning(f"فشل حساب مقاييس ROC AUC: {str(e)}")
                
                # منحنى المعايرة
                try:
                    prob_true, prob_pred = calibration_curve(y, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1))
                    metrics['calibration_curve'] = {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
                except Exception as e:
                    logger.warning(f"فشل حساب منحنى المعايرة: {str(e)}")
            
            # مصفوفة الارتباك وتقرير التصنيف
            cm = confusion_matrix(y, y_pred, sample_weight=sample_weight)
            metrics['confusion_matrix'] = cm.tolist()
            
            class_report = classification_report(y, y_pred, output_dict=True, sample_weight=sample_weight)
            metrics['classification_report'] = class_report
            
            return metrics
            
        except Exception as e:
            raise ModelEvaluationError(f"خطأ في تقييم نموذج التصنيف: {str(e)}")
            
    async def _evaluate_regression(self,
                           model: ModelType,
                           X: ArrayLike,
                           y: ArrayLike,
                           sample_weight: Optional[ArrayLike] = None) -> Dict[str, float]:
        """تقييم نموذج الانحدار مع مقاييس متقدمة"""
        try:
            y_pred = model.predict(X)
            metrics = {
                'mse': mean_squared_error(y, y_pred, sample_weight=sample_weight),
                'rmse': np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight)),
                'mae': mean_absolute_error(y, y_pred, sample_weight=sample_weight),
                'r2': r2_score(y, y_pred, sample_weight=sample_weight),
                'explained_variance': explained_variance_score(y, y_pred, sample_weight=sample_weight),
                'max_error': max_error(y, y_pred),
                'mape': mean_absolute_percentage_error(y, y_pred)
            }
            
            # تحليل الأخطاء
            residuals = y - y_pred
            metrics['residuals_stats'] = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skew': float(pd.Series(residuals).skew()),
                'kurtosis': float(pd.Series(residuals).kurtosis())
            }
            
            # تحليل الأخطاء حسب النطاق
            percentiles = np.percentile(y, [25, 50, 75])
            for i, (lower, upper) in enumerate(zip([-np.inf] + list(percentiles), list(percentiles) + [np.inf])):
                mask = (y > lower) & (y <= upper)
                if np.any(mask):
                    metrics[f'rmse_range_{i+1}'] = float(
                        np.sqrt(mean_squared_error(y[mask], y_pred[mask], sample_weight=sample_weight[mask] if sample_weight is not None else None))
                    )
            
            return metrics
            
        except Exception as e:
            raise ModelEvaluationError(f"خطأ في تقييم نموذج الانحدار: {str(e)}")
            
    async def _evaluate_clustering(self,
                           model: ModelType,
                           X: np.ndarray) -> Dict[str, float]:
        """تقييم نموذج التجميع مع مقاييس متقدمة"""
        try:
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            
            metrics = {
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels)
            }
            
            # إحصائيات المجموعات
            unique_labels = np.unique(labels)
            cluster_stats = {}
            
            for label in unique_labels:
                mask = labels == label
                cluster_points = X[mask]
                
                cluster_stats[f'cluster_{label}'] = {
                    'size': int(np.sum(mask)),
                    'density': float(np.mean(np.linalg.norm(cluster_points - np.mean(cluster_points, axis=0), axis=1))),
                    'mean': cluster_points.mean(axis=0).tolist(),
                    'std': cluster_points.std(axis=0).tolist()
                }
                
            metrics['cluster_stats'] = cluster_stats
            
            return metrics
            
        except Exception as e:
            raise ModelEvaluationError(f"خطأ في تقييم نموذج التجميع: {str(e)}")
            
    async def _calculate_feature_importance(self,
                                   model: ModelType,
                                   X: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        """حساب أهمية الميزات مع دعم لأنواع مختلفة من النماذج"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return {}
                
            for name, score in zip(feature_names, importance_scores):
                importance_dict[name] = float(score)
                
            # تطبيع القيم
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v/total for k, v in importance_dict.items()}
                
            return importance_dict
            
        except Exception as e:
            logger.warning(f"فشل حساب أهمية الميزات: {str(e)}")
            return {}
            
    async def _calculate_shap_values(self,
                             model: ModelType,
                             X: np.ndarray,
                             feature_names: List[str],
                             sample_size: int = 100) -> Dict[str, Any]:
        """حساب قيم SHAP مع تحسينات"""
        try:
            # اختيار عينة عشوائية للحسابات الكبيرة
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            if self.task_type in ['classification', 'regression']:
                # اختيار نوع المفسر المناسب
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
                
                self.shap_values = explainer.shap_values(X_sample)
                
                # تنظيم النتائج
                shap_dict = {
                    'values': self.shap_values if isinstance(self.shap_values, np.ndarray) 
                             else np.array(self.shap_values),
                    'feature_names': feature_names,
                    'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None,
                    'interaction_values': None  # يمكن إضافة حساب قيم التفاعل إذا لزم الأمر
                }
                
                return shap_dict
                
            return {}
            
        except Exception as e:
            logger.warning(f"فشل حساب قيم SHAP: {str(e)}")
            return {}
            
    async def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """حفظ نتائج التقييم"""
        try:
            # إضافة إلى التاريخ
            self.evaluation_history.append(results)
            
            # حفظ في ملف
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_file = self.evaluation_dir / f"evaluation_{timestamp}.joblib"
            joblib.dump(results, eval_file)
            
        except Exception as e:
            logger.warning(f"فشل حفظ نتائج التقييم: {str(e)}")
            
    def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        """الحصول على آخر نتائج تقييم"""
        return self.evaluation_history[-1] if self.evaluation_history else None

model_evaluator = ModelEvaluator() 