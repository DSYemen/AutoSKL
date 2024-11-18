from typing import Dict, Any, List, Optional, Tuple, Protocol, TypeVar
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    LinearRegression, Ridge, Lasso,
    ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, Birch, MiniBatchKMeans
)
from sklearn.decomposition import (
    PCA, TruncatedSVD, FastICA,
    NMF, LatentDirichletAllocation
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from app.core.config import settings
from app.utils.exceptions import ModelSelectionError
import logging
from datetime import datetime
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ModelType = TypeVar('ModelType', bound=BaseEstimator)
ArrayLike: TypeAlias = np.ndarray
ModelParams: TypeAlias = Dict[str, Any]

class ModelProtocol(Protocol):
    """بروتوكول للنماذج"""
    def fit(self, X: ArrayLike, y: ArrayLike) -> None: ...
    def predict(self, X: ArrayLike) -> ArrayLike: ...
    def set_params(self, **params: Any) -> None: ...

class ModelSelector:
    """محدد النماذج"""
    def __init__(self) -> None:
        self.task_type: Optional[str] = None
        self.best_model: Optional[ModelType] = None
        self.best_params: Optional[ModelParams] = None
        self.best_score: float = float('-inf')
        self.cv_results: List[Dict[str, Any]] = []
        self.study: Optional[optuna.Study] = None
        
    async def select_best_model(self,
                              X: ArrayLike,
                              y: ArrayLike,
                              task_type: str,
                              training_params: Optional[ModelParams] = None) -> Tuple[ModelType, ModelParams]:
        """اختيار أفضل نموذج باستخدام Optuna"""
        try:
            self.task_type = task_type
            
            # إنشاء دراسة Optuna
            self.study = optuna.create_study(
                direction="maximize",
                study_name=f"{task_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # تحديد دالة الهدف
            def objective(trial: optuna.Trial) -> float:
                # اختيار نوع النموذج
                model_type = trial.suggest_categorical('model_type', list(self.get_models().keys()))
                
                # الحصول على نطاقات المعلمات للنموذج المحدد
                param_ranges = self._get_param_ranges(model_type, trial)
                
                # إنشاء النموذج مع المعلمات المقترحة
                model = self.get_models()[model_type]
                model.set_params(**param_ranges)
                
                # التحقق المتقاطع
                cv = StratifiedKFold(n_splits=settings.ml.model_selection.cv_folds) if task_type == 'classification' else KFold(n_splits=settings.ml.model_selection.cv_folds)
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=settings.ml.model_selection.metric[task_type],
                    n_jobs=-1
                )
                
                return scores.mean()
                
            # تنفيذ التحسين
            self.study.optimize(
                objective,
                n_trials=settings.ml.model_selection.n_trials,
                timeout=settings.ml.model_selection.timeout,
                callbacks=[self._optimization_callback],
                n_jobs=1  # لتجنب مشاكل التزامن مع async
            )
            
            # تدريب النموذج النهائي
            best_params = self.study.best_params
            model_type = best_params.pop('model_type')
            best_model = self.get_models()[model_type]
            best_model.set_params(**best_params)
            best_model.fit(X, y)
            
            self.best_model = best_model
            self.best_params = {
                'model_type': model_type,
                **best_params
            }
            
            logger.info(f"تم اختيار أفضل نموذج: {model_type} مع معلمات: {best_params}")
            return best_model, self.best_params
            
        except Exception as e:
            logger.error(f"خطأ في اختيار النموذج: {str(e)}")
            raise ModelSelectionError(f"فشل اختيار النموذج: {str(e)}")
            
    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """معالج استدعاء التحسين"""
        try:
            # تحديث أفضل نتيجة
            if trial.value > self.best_score:
                self.best_score = trial.value
                
            # تسجيل النتائج
            self.cv_results.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'datetime': datetime.now().isoformat()
            })
            
            # التوقف المبكر إذا تم تمكينه
            early_stopping_config = settings.ml.model_selection.early_stopping
            if early_stopping_config.get('enabled', True):
                patience = early_stopping_config.get('patience', 10)
                min_delta = early_stopping_config.get('min_delta', 0.001)
                
                last_n_trials = study.trials[-patience:]
                if len(last_n_trials) == patience:
                    values = [t.value for t in last_n_trials if t.value is not None]
                    if len(values) > 0:
                        best_value = max(values)
                        if all(abs(v - best_value) < min_delta for v in values):
                            study.stop()
                            
        except Exception as e:
            logger.warning(f"خطأ في معالج استدعاء التحسين: {str(e)}")
            
    def _get_param_ranges(self, model_type: str, trial: optuna.Trial) -> Dict[str, Any]:
        """الحصول على نطاقات المعلمات للنموذج"""
        param_ranges = {}
        
        if 'random_forest' in model_type:
            param_ranges.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            })
        elif 'xgboost' in model_type:
            param_ranges.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            })
        elif 'lightgbm' in model_type:
            param_ranges.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0)
            })
        elif 'catboost' in model_type:
            param_ranges.update({
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            })
        elif 'svm' in model_type.lower():
            param_ranges.update({
                'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            })
            
        return param_ranges
        
    def get_models(self) -> Dict[str, BaseEstimator]:
        """الحصول على قائمة النماذج المتاحة حسب نوع المهمة"""
        if self.task_type == 'classification':
            return {
                'logistic_regression': LogisticRegression(max_iter=1000),
                'random_forest': RandomForestClassifier(),
                'gradient_boosting': GradientBoostingClassifier(),
                'hist_gradient_boosting': HistGradientBoostingClassifier(),
                'extra_trees': ExtraTreesClassifier(),
                'ada_boost': AdaBoostClassifier(),
                'svc': SVC(probability=True),
                'linear_svc': LinearSVC(max_iter=1000),
                'knn': KNeighborsClassifier(),
                'decision_tree': DecisionTreeClassifier(),
                'mlp': MLPClassifier(max_iter=1000),
                'gaussian_nb': GaussianNB(),
                'multinomial_nb': MultinomialNB(),
                'lda': LinearDiscriminantAnalysis(),
                'ridge': RidgeClassifier(),
                'sgd': SGDClassifier(max_iter=1000),
                'xgboost': XGBClassifier(tree_method='hist'),
                'lightgbm': LGBMClassifier(),
                'catboost': CatBoostClassifier(verbose=False)
            }
        elif self.task_type == 'regression':
            return {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'elastic_net': ElasticNet(),
                'random_forest': RandomForestRegressor(),
                'gradient_boosting': GradientBoostingRegressor(),
                'hist_gradient_boosting': HistGradientBoostingRegressor(),
                'extra_trees': ExtraTreesRegressor(),
                'ada_boost': AdaBoostRegressor(),
                'svr': SVR(),
                'linear_svr': LinearSVR(max_iter=1000),
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor(),
                'mlp': MLPRegressor(max_iter=1000),
                'sgd': SGDRegressor(max_iter=1000),
                'xgboost': XGBRegressor(tree_method='hist'),
                'lightgbm': LGBMRegressor(),
                'catboost': CatBoostRegressor(verbose=False)
            }
        elif self.task_type == 'clustering':
            return {
                'kmeans': KMeans(),
                'dbscan': DBSCAN(),
                'hierarchical': AgglomerativeClustering(),
                'spectral': SpectralClustering(),
                'birch': Birch(),
                'minibatch_kmeans': MiniBatchKMeans()
            }
        elif self.task_type == 'dimensionality_reduction':
            return {
                'pca': PCA(),
                'truncated_svd': TruncatedSVD(),
                'fast_ica': FastICA(),
                'nmf': NMF(),
                'lda': LatentDirichletAllocation()
            }
        else:
            raise ModelSelectionError(f"نوع المهمة غير معروف: {self.task_type}")
            
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """الحصول على تاريخ التحسين"""
        return self.cv_results
        
    def get_best_trial_info(self) -> Dict[str, Any]:
        """الحصول على معلومات أفضل تجربة"""
        if self.study is None:
            return {}
            
        return {
            'value': self.study.best_value,
            'params': self.study.best_params,
            'trial_number': self.study.best_trial.number,
            'datetime': self.study.best_trial.datetime.isoformat()
        }

model_selector = ModelSelector() 