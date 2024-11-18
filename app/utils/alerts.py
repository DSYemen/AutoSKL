from typing import Dict, Any, List, Optional, Protocol, TypeVar
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import logging
from app.core.config import settings
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import aiosmtplib
import asyncio
from functools import lru_cache
import json
import pandas as pd
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
AlertData: TypeAlias = Dict[str, Any]
AlertHistory: TypeAlias = List[Dict[str, Any]]

class AlertManager:
    """مدير التنبيهات"""
    def __init__(self) -> None:
        self.template_dir = Path("app/templates/alerts")
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        self.email_config = settings.email
        self.alert_history: AlertHistory = []
        self.last_alert_times: Dict[str, datetime] = {}
        
    async def send_alert(self,
                        subject: str,
                        template_name: str,
                        data: AlertData,
                        recipients: List[str],
                        attachments: Optional[List[str]] = None) -> None:
        """إرسال تنبيه عبر البريد الإلكتروني بشكل غير متزامن"""
        try:
            # التحقق من فترة التهدئة
            alert_key = f"{template_name}_{json.dumps(data, sort_keys=True)}"
            if not self._check_cooldown(alert_key):
                logger.info(f"تم تجاهل التنبيه بسبب فترة التهدئة: {subject}")
                return
                
            # إعداد رسالة البريد
            msg = MIMEMultipart()
            msg['Subject'] = f"[{settings.app.name}] {subject}"
            msg['From'] = self.email_config.sender
            msg['To'] = ", ".join(recipients)
            
            # توليد محتوى البريد
            template = self.template_env.get_template(f"{template_name}.html")
            html_content = template.render(
                **data,
                app_name=settings.app.name,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # إضافة المرفقات
            if attachments:
                for file_path in attachments:
                    with open(file_path, 'rb') as f:
                        part = MIMEApplication(f.read(), Name=Path(file_path).name)
                        part['Content-Disposition'] = f'attachment; filename="{Path(file_path).name}"'
                        msg.attach(part)
            
            # إرسال البريد بشكل غير متزامن
            await self._send_email_async(msg)
            
            # تحديث سجل التنبيهات
            self._update_alert_history(subject, template_name, data, recipients)
            
            logger.info(f"تم إرسال التنبيه بنجاح إلى {recipients}")
            
        except Exception as e:
            logger.error(f"خطأ في إرسال التنبيه: {str(e)}")
            raise
            
    async def _send_email_async(self, msg: MIMEMultipart) -> None:
        """إرسال البريد الإلكتروني بشكل غير متزامن"""
        try:
            async with aiosmtplib.SMTP(
                hostname=self.email_config.smtp_server,
                port=self.email_config.smtp_port,
                use_tls=self.email_config.use_tls,
                timeout=self.email_config.timeout
            ) as server:
                if self.email_config.use_tls:
                    await server.starttls()
                await server.login(self.email_config.sender, self.email_config.password)
                await server.send_message(msg)
        except Exception as e:
            logger.error(f"خطأ في إرسال البريد الإلكتروني: {str(e)}")
            raise
            
    def _check_cooldown(self, alert_key: str) -> bool:
        """التحقق من فترة التهدئة للتنبيه"""
        now = datetime.utcnow()
        if alert_key in self.last_alert_times:
            time_diff = (now - self.last_alert_times[alert_key]).total_seconds()
            if time_diff < settings.monitoring.alert_cooldown:
                return False
        self.last_alert_times[alert_key] = now
        return True
        
    def _update_alert_history(self,
                            subject: str,
                            template_name: str,
                            data: AlertData,
                            recipients: List[str]) -> None:
        """تحديث سجل التنبيهات"""
        self.alert_history.append({
            'subject': subject,
            'template': template_name,
            'data': data,
            'recipients': recipients,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # الاحتفاظ بآخر 1000 تنبيه فقط
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
            
    async def send_model_performance_alert(self,
                                         model_id: str,
                                         metrics: Dict[str, float],
                                         threshold: float,
                                         recipients: List[str]) -> None:
        """إرسال تنبيه أداء النموذج"""
        data = {
            'model_id': model_id,
            'metrics': metrics,
            'threshold': threshold,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        await self.send_alert(
            subject=f"تنبيه أداء النموذج - {model_id}",
            template_name='model_performance_alert',
            data=data,
            recipients=recipients
        )
        
    async def send_data_drift_alert(self,
                                  model_id: str,
                                  drift_metrics: Dict[str, Any],
                                  recipients: List[str]) -> None:
        """إرسال تنبيه انحراف البيانات"""
        data = {
            'model_id': model_id,
            'drift_metrics': drift_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        await self.send_alert(
            subject=f"تنبيه انحراف البيانات - {model_id}",
            template_name='data_drift_alert',
            data=data,
            recipients=recipients
        )
        
    async def send_error_alert(self,
                             error_type: str,
                             error_message: str,
                             model_id: Optional[str],
                             recipients: List[str]) -> None:
        """إرسال تنبيه خطأ"""
        data = {
            'error_type': error_type,
            'error_message': error_message,
            'model_id': model_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        await self.send_alert(
            subject=f"تنبيه خطأ - {error_type}",
            template_name='error_alert',
            data=data,
            recipients=recipients
        )
        
    async def send_model_update_alert(self,
                                    model_id: str,
                                    old_performance: Dict[str, float],
                                    new_performance: Dict[str, float],
                                    recipients: List[str]) -> None:
        """إرسال تنبيه تحديث النموذج"""
        data = {
            'model_id': model_id,
            'old_performance': old_performance,
            'new_performance': new_performance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        await self.send_alert(
            subject=f"تنبيه تحديث النموذج - {model_id}",
            template_name='model_update_alert',
            data=data,
            recipients=recipients
        )
        
    async def send_startup_alert(self) -> None:
        """إرسال تنبيه بدء التشغيل"""
        data = {
            'app_version': settings.app.version,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # استخدام القيمة الافتراضية إذا لم تكن alert_recipients موجودة
        recipients = (settings.email.get('alert_recipients', []) 
                     if isinstance(settings.email, dict) 
                     else getattr(settings.email, 'alert_recipients', []))
        
        if recipients:
            await self.send_alert(
                subject="تنبيه بدء تشغيل النظام",
                template_name='startup_alert',
                data=data,
                recipients=recipients
            )
        
    async def send_shutdown_alert(self) -> None:
        """إرسال تنبيه إيقاف التشغيل"""
        data = {
            'app_version': settings.app.version,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        await self.send_alert(
            subject="تنبيه إيقاف تشغيل النظام",
            template_name='shutdown_alert',
            data=data,
            recipients=settings.email.alert_recipients
        )
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبيهات"""
        if not self.alert_history:
            return {}
            
        alerts_df = pd.DataFrame(self.alert_history)
        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_type': alerts_df['template'].value_counts().to_dict(),
            'last_alert_time': alerts_df['timestamp'].max(),
            'most_common_recipients': pd.Series([
                r for recipients in alerts_df['recipients'] for r in recipients
            ]).value_counts().head(5).to_dict()
        }

alert_manager = AlertManager() 