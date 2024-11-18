from typing import Dict, Any, List, Optional, Protocol, TypeVar
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import pdfkit
from pathlib import Path
from app.utils.visualization import visualization_manager
from app.core.config import settings
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
ReportData: TypeAlias = Dict[str, Any]
PlotResult: TypeAlias = Dict[str, str]

class ReportGenerator:
    """مولد التقارير"""
    def __init__(self) -> None:
        self.template_dir = Path(settings.reporting.template_dir)
        self.output_dir = Path(settings.storage.reports_dir)
        self.template_loader = jinja2.FileSystemLoader(searchpath=str(self.template_dir))
        self.template_env = jinja2.Environment(
            loader=self.template_loader,
            autoescape=True
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # إنشاء المجلدات المطلوبة
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
    async def generate_model_report(self,
                                  model_id: str,
                                  model_info: Dict[str, Any],
                                  evaluation_results: Dict[str, Any],
                                  feature_importance: Dict[str, float],
                                  predictions_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """توليد تقرير شامل عن النموذج"""
        try:
            # إعداد بيانات التقرير
            report_data = {
                'model_id': model_id,
                'model_info': model_info,
                'evaluation_results': evaluation_results,
                'feature_importance': feature_importance,
                'predictions_history': predictions_history,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # إنشاء المخططات بشكل متوازي
            plots = await self._create_report_plots_async(report_data)
            report_data['plots'] = plots
            
            # توليد HTML
            template = self.template_env.get_template('model_report.html')
            html_content = template.render(**report_data)
            
            # تحويل HTML إلى PDF
            output_path = self.output_dir / f"model_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # استخدام ThreadPoolExecutor لتحويل PDF
            await self._convert_to_pdf_async(html_content, str(output_path))
            
            logger.info(f"تم إنشاء تقرير النموذج في {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء تقرير النموذج: {str(e)}")
            raise
            
    async def _create_report_plots_async(self, report_data: ReportData) -> PlotResult:
        """إنشاء المخططات بشكل متوازي"""
        plots = {}
        tasks = []
        
        # إنشاء مخطط أهمية الميزات
        if report_data['feature_importance']:
            tasks.append(self._create_plot_async(
                'feature_importance',
                lambda: visualization_manager.create_feature_importance_plot(
                    pd.Series(report_data['feature_importance']),
                    interactive=False
                )
            ))
            
        # إنشاء مخطط مصفوفة الارتباك للتصنيف
        if 'confusion_matrix' in report_data['evaluation_results']:
            tasks.append(self._create_plot_async(
                'confusion_matrix',
                lambda: visualization_manager.create_confusion_matrix_plot(
                    np.array(report_data['evaluation_results']['confusion_matrix']),
                    interactive=False
                )
            ))
            
        # إنشاء مخطط تاريخ الأداء
        if report_data['predictions_history']:
            tasks.append(self._create_plot_async(
                'performance_history',
                lambda: self._create_performance_history_plot(
                    report_data['predictions_history']
                )
            ))
            
        # انتظار اكتمال جميع المهام
        completed_plots = await asyncio.gather(*tasks)
        
        # تجميع النتائج
        for plot_name, plot_path in completed_plots:
            plots[plot_name] = plot_path
            
        return plots
        
    async def _create_plot_async(self, name: str, plot_func) -> tuple[str, str]:
        """إنشاء مخطط بشكل غير متزامن"""
        try:
            # تنفيذ دالة إنشاء المخطط في مجمع الخيوط
            plot_path = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                plot_func
            )
            return name, plot_path
        except Exception as e:
            logger.error(f"خطأ في إنشاء مخطط {name}: {str(e)}")
            return name, ""
            
    async def _convert_to_pdf_async(self, html_content: str, output_path: str) -> None:
        """تحويل HTML إلى PDF بشكل غير متزامن"""
        try:
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': 'UTF-8',
                'custom-header': [
                    ('Accept-Encoding', 'gzip')
                ],
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: pdfkit.from_string(html_content, output_path, options=options)
            )
            
        except Exception as e:
            logger.error(f"خطأ في تحويل HTML إلى PDF: {str(e)}")
            raise
            
    def _create_performance_history_plot(self, history: List[Dict[str, Any]]) -> go.Figure:
        """إنشاء مخطط تاريخ الأداء"""
        df = pd.DataFrame(history)
        
        fig = make_subplots(rows=2, cols=1)
        
        # إضافة مقاييس الأداء
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        name=metric.capitalize()
                    ),
                    row=1, col=1
                )
                
        # إضافة عدد التنبؤات
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['predictions_count'],
                name='Predictions Count'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Model Performance History',
            height=800,
            showlegend=True,
            template=settings.reporting.plots.theme
        )
        
        return fig

report_generator = ReportGenerator() 