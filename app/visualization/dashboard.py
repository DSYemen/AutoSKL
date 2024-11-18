from typing import Dict, Any, List, Optional, Protocol, TypeVar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.utils.visualization import visualization_manager
from app.core.config import settings
import logging
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
DashboardData: TypeAlias = Dict[str, Any]
MetricsData: TypeAlias = List[Dict[str, Any]]

class Dashboard:
    """لوحة المعلومات"""
    def __init__(self) -> None:
        self.theme = settings.reporting.plots.theme
        self.default_height = settings.reporting.plots.height
        self.default_width = settings.reporting.plots.width
        
    async def create_performance_dashboard(self,
                                         model_metrics: MetricsData,
                                         drift_data: List[Dict[str, Any]],
                                         prediction_stats: Dict[str, Any]) -> go.Figure:
        """إنشاء لوحة معلومات الأداء"""
        try:
            # إنشاء تخطيط الشبكة
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Model Performance Over Time',
                    'Data Drift Detection',
                    'Prediction Distribution',
                    'Feature Importance',
                    'Error Analysis',
                    'System Metrics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "heatmap"}],
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "indicator"}, {"type": "scatter"}]
                ]
            )
            
            # إضافة مخطط الأداء عبر الزمن
            await self._add_performance_plot(fig, model_metrics, row=1, col=1)
            
            # إضافة مخطط اكتشاف الانحراف
            await self._add_drift_plot(fig, drift_data, row=1, col=2)
            
            # إضافة مخطط توزيع التنبؤات
            await self._add_prediction_distribution(fig, prediction_stats, row=2, col=1)
            
            # إضافة مخطط أهمية الميزات
            await self._add_feature_importance(fig, model_metrics[-1], row=2, col=2)
            
            # إضافة تحليل الأخطاء
            await self._add_error_analysis(fig, model_metrics, row=3, col=1)
            
            # إضافة مؤشرات النظام
            await self._add_system_metrics(fig, row=3, col=2)
            
            # تحديث تخطيط اللوحة
            fig.update_layout(
                height=self.default_height * 2,
                width=self.default_width * 2,
                showlegend=True,
                template=self.theme,
                title={
                    'text': 'Model Performance Dashboard',
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء لوحة المعلومات: {str(e)}")
            raise
            
    async def _add_performance_plot(self,
                                  fig: go.Figure,
                                  metrics: MetricsData,
                                  row: int,
                                  col: int) -> None:
        """إضافة مخطط الأداء"""
        try:
            df = pd.DataFrame(metrics)
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            
            for metric in metrics_to_plot:
                if metric in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[metric],
                            name=metric.capitalize(),
                            mode='lines+markers'
                        ),
                        row=row, col=col
                    )
                    
            fig.update_xaxes(title_text="Time", row=row, col=col)
            fig.update_yaxes(title_text="Score", row=row, col=col)
            
        except Exception as e:
            logger.error(f"خطأ في إضافة مخطط الأداء: {str(e)}")
            
    async def _add_drift_plot(self,
                            fig: go.Figure,
                            drift_data: List[Dict[str, Any]],
                            row: int,
                            col: int) -> None:
        """إضافة مخطط اكتشاف الانحراف"""
        try:
            df = pd.DataFrame(drift_data)
            if not df.empty:
                drift_matrix = pd.pivot_table(
                    df,
                    values='drift_score',
                    index='feature_name',
                    columns='detection_time',
                    aggfunc='first'
                ).fillna(0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=drift_matrix.values,
                        x=drift_matrix.columns,
                        y=drift_matrix.index,
                        colorscale='RdBu',
                        showscale=True
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.error(f"خطأ في إضافة مخطط الانحراف: {str(e)}")
            
    async def _add_prediction_distribution(self,
                                        fig: go.Figure,
                                        stats: Dict[str, Any],
                                        row: int,
                                        col: int) -> None:
        """إضافة مخطط توزيع التنبؤات"""
        try:
            if 'predictions' in stats:
                predictions = stats['predictions']
                fig.add_trace(
                    go.Histogram(
                        x=predictions,
                        nbinsx=30,
                        name='Predictions'
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.error(f"خطأ في إضافة توزيع التنبؤات: {str(e)}")
            
    async def _add_feature_importance(self,
                                   fig: go.Figure,
                                   metrics: Dict[str, Any],
                                   row: int,
                                   col: int) -> None:
        """إضافة مخطط أهمية الميزات"""
        try:
            if 'feature_importance' in metrics:
                importance = pd.Series(metrics['feature_importance'])
                importance = importance.sort_values(ascending=True)
                
                fig.add_trace(
                    go.Bar(
                        x=importance.values,
                        y=importance.index,
                        orientation='h',
                        name='Feature Importance'
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.error(f"خطأ في إضافة أهمية الميزات: {str(e)}")
            
    async def _add_system_metrics(self,
                               fig: go.Figure,
                               row: int,
                               col: int) -> None:
        """إضافة مؤشرات النظام"""
        try:
            # إضافة مؤشرات النظام (يمكن تحديثها حسب احتياجاتك)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=70,  # قيمة مثال
                    title={'text': "System Health"},
                    gauge={'axis': {'range': [0, 100]}},
                    domain={'row': row-1, 'column': col-1}
                )
            )
            
        except Exception as e:
            logger.error(f"خطأ في إضافة مؤشرات النظام: {str(e)}")
            
    async def _add_error_analysis(self,
                                fig: go.Figure,
                                metrics: MetricsData,
                                row: int,
                                col: int) -> None:
        """إضافة تحليل الأخطاء"""
        try:
            df = pd.DataFrame(metrics)
            if 'error_distribution' in df.columns:
                error_dist = df['error_distribution'].iloc[-1]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(error_dist.keys()),
                        y=list(error_dist.values()),
                        mode='lines+markers',
                        name='Error Distribution'
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            logger.error(f"خطأ في إضافة تحليل الأخطاء: {str(e)}")
            
    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """الحصول على إحصائيات ملخصة"""
        try:
            return {
                'drift_detected_count': df['drift_detected'].sum(),
                'features_affected': df[df['drift_detected']]['feature_name'].nunique(),
                'last_detection': df['detection_time'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"خطأ في حساب الإحصائيات الملخصة: {str(e)}")
            return {}

dashboard = Dashboard() 