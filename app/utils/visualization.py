from typing import Dict, Any, List, Optional, Protocol, TypeVar, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
Figure = TypeVar('Figure', bound=go.Figure)
ArrayLike: TypeAlias = np.ndarray | pd.Series | List[float]
PlotResult: TypeAlias = Union[go.Figure, str]

class VisualizationManager:
    """مدير التصور المرئي"""
    def __init__(self) -> None:
        self.default_theme = "plotly_dark"
        self.default_colors = px.colors.qualitative.Set3
        self.output_dir = Path("static/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_feature_importance_plot(self,
                                     importance_scores: pd.Series,
                                     title: str = "Feature Importance",
                                     top_n: Optional[int] = None,
                                     interactive: bool = True) -> PlotResult:
        """إنشاء مخطط محسن لأهمية الميزات"""
        try:
            sorted_scores = importance_scores.sort_values(ascending=True)
            if top_n:
                sorted_scores = sorted_scores.tail(top_n)
                
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=sorted_scores.values,
                    y=sorted_scores.index,
                    orientation='h',
                    marker_color=self.default_colors[0]
                )
            )
            
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=max(400, len(sorted_scores) * 20),
                template=self.default_theme,
                showlegend=False
            )
            
            if interactive:
                return fig
            else:
                output_path = self.output_dir / f"feature_importance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                fig.write_image(str(output_path))
                return str(output_path)
                
        except Exception as e:
            logger.error(f"خطأ في إنشاء مخطط أهمية الميزات: {str(e)}")
            raise
            
    def create_confusion_matrix_plot(self,
                                   y_true: ArrayLike,
                                   y_pred: ArrayLike,
                                   labels: Optional[List[str]] = None,
                                   title: str = "Confusion Matrix",
                                   normalize: bool = False) -> PlotResult:
        """إنشاء مخطط محسن لمصفوفة الارتباك"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
            if labels is None:
                labels = [str(i) for i in range(len(cm))]
                
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='RdBu',
                    text=np.around(cm, decimals=2),
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    hoverongaps=False
                )
            )
            
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                template=self.default_theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء مخطط مصفوفة الارتباك: {str(e)}")
            raise
            
    def create_learning_curve_plot(self,
                                 train_sizes: ArrayLike,
                                 train_scores: ArrayLike,
                                 val_scores: ArrayLike,
                                 title: str = "Learning Curve",
                                 metric_name: str = "Score") -> PlotResult:
        """إنشاء مخطط محسن لمنحنى التعلم"""
        try:
            fig = go.Figure()
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    name="Training Score",
                    mode="lines+markers",
                    line=dict(color=self.default_colors[0]),
                    error_y=dict(
                        type='data',
                        array=train_std,
                        visible=True
                    )
                )
            )
            
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=val_mean,
                    name="Validation Score",
                    mode="lines+markers",
                    line=dict(color=self.default_colors[1]),
                    error_y=dict(
                        type='data',
                        array=val_std,
                        visible=True
                    )
                )
            )
            
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Training Examples",
                yaxis_title=metric_name,
                template=self.default_theme,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء مخطط منحنى التعلم: {str(e)}")
            raise
            
    def create_prediction_distribution_plot(self,
                                          predictions: ArrayLike,
                                          actual: Optional[ArrayLike] = None,
                                          title: str = "Prediction Distribution",
                                          bins: int = 30) -> PlotResult:
        """إنشاء مخطط محسن لتوزيع التنبؤات"""
        try:
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=predictions,
                    name="Predictions",
                    nbinsx=bins,
                    opacity=0.7,
                    marker_color=self.default_colors[0]
                )
            )
            
            if actual is not None:
                fig.add_trace(
                    go.Histogram(
                        x=actual,
                        name="Actual",
                        nbinsx=bins,
                        opacity=0.7,
                        marker_color=self.default_colors[1]
                    )
                )
                
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Value",
                yaxis_title="Count",
                template=self.default_theme,
                showlegend=True,
                barmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء مخطط توزيع التنبؤات: {str(e)}")
            raise

visualization_manager = VisualizationManager()