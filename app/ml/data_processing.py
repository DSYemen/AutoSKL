from typing import Tuple, Dict, Any, Optional, List, Protocol
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    VarianceThreshold, RFE, mutual_info_classif, mutual_info_regression
)
from app.core.config import settings
from app.utils.exceptions import DataProcessingError
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """معالج البيانات"""
    def __init__(self) -> None:
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.target_column: Optional[str] = None
        self.task_type: Optional[str] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        
    async def process_data(self, 
                          df: pd.DataFrame, 
                          target_column: str, 
                          task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """معالجة البيانات"""
        try:
            self.target_column = target_column
            self.task_type = task_type
            
            # التحقق من صحة البيانات
            self._validate_data(df)
            
            # تحديد أنواع الأعمدة
            self._identify_feature_types(df)
            
            # إنشاء معالجات البيانات
            numeric_transformer = Pipeline(steps=[
                ('imputer', IterativeImputer(random_state=42)),
                ('scaler', RobustScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False))
            ])
            
            # إنشاء المعالج الرئيسي
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ],
                remainder='drop'
            )
            
            # معالجة البيانات
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # تطبيق المعالجة
            X_processed = self.preprocessor.fit_transform(X)
            
            # معالجة المتغير الهدف
            if task_type == 'classification':
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y.values
                
            return X_processed, y_processed
            
        except Exception as e:
            logger.error(f"خطأ في معالجة البيانات: {str(e)}")
            raise DataProcessingError(f"خطأ في معالجة البيانات: {str(e)}")
            
    def _validate_data(self, df: pd.DataFrame) -> None:
        """التحقق من صحة البيانات"""
        if df.empty:
            raise DataProcessingError("البيانات فارغة")
            
        if self.target_column not in df.columns:
            raise DataProcessingError(f"عمود الهدف {self.target_column} غير موجود")
            
    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """تحديد أنواع الميزات"""
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.numeric_features = df.select_dtypes(include=numeric_dtypes).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # إزالة عمود الهدف من قوائم الميزات
        for features in [self.numeric_features, self.categorical_features]:
            if self.target_column in features:
                features.remove(self.target_column)

data_processor = DataProcessor() 