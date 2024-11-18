import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from app.utils.alerts import alert_manager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@pytest.fixture
def mock_smtp():
    """نموذج SMTP وهمي"""
    with patch('smtplib.SMTP') as mock:
        yield mock

@pytest.fixture
def sample_data():
    """بيانات نموذجية للتنبيهات"""
    return {
        'model_id': 'test_model',
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.83
        },
        'threshold': 0.8,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def test_send_alert(mock_smtp, sample_data):
    """اختبار إرسال تنبيه"""
    recipients = ['test@example.com']
    
    alert_manager.send_alert(
        'Test Alert',
        'model_performance_alert',
        sample_data,
        recipients
    )
    
    # التحقق من إنشاء اتصال SMTP
    mock_smtp.return_value.__enter__.return_value.starttls.assert_called_once()
    mock_smtp.return_value.__enter__.return_value.login.assert_called_once()
    mock_smtp.return_value.__enter__.return_value.send_message.assert_called_once()

def test_send_model_performance_alert(mock_smtp):
    """اختبار إرسال تنبيه أداء النموذج"""
    metrics = {
        'accuracy': 0.85,
        'precision': 0.83
    }
    
    alert_manager.send_model_performance_alert(
        'test_model',
        metrics,
        0.8,
        ['test@example.com']
    )
    
    # التحقق من إرسال البريد
    mock_smtp.return_value.__enter__.return_value.send_message.assert_called_once()
    
    # التحقق من محتوى البريد
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args[0][0]
    assert 'تنبيه أداء النموذج' in sent_message['Subject']
    assert 'test_model' in sent_message['Subject']

def test_send_data_drift_alert(mock_smtp):
    """اختبار إرسال تنبيه انحراف البيانات"""
    drift_metrics = {
        'feature1': {
            'drift_score': 0.1,
            'p_value': 0.05,
            'drift_detected': True
        }
    }
    
    alert_manager.send_data_drift_alert(
        'test_model',
        drift_metrics,
        ['test@example.com']
    )
    
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args[0][0]
    assert 'تنبيه انحراف البيانات' in sent_message['Subject']

def test_send_error_alert(mock_smtp):
    """اختبار إرسال تنبيه خطأ"""
    alert_manager.send_error_alert(
        'test_error',
        'خطأ في النظام',
        'test_model',
        ['test@example.com']
    )
    
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args[0][0]
    assert 'تنبيه خطأ' in sent_message['Subject']
    assert 'test_error' in sent_message['Subject']

def test_send_model_update_alert(mock_smtp):
    """اختبار إرسال تنبيه تحديث النموذج"""
    old_performance = {'accuracy': 0.8}
    new_performance = {'accuracy': 0.85}
    
    alert_manager.send_model_update_alert(
        'test_model',
        old_performance,
        new_performance,
        ['test@example.com']
    )
    
    sent_message = mock_smtp.return_value.__enter__.return_value.send_message.call_args[0][0]
    assert 'تنبيه تحديث النموذج' in sent_message['Subject']

def test_template_rendering():
    """اختبار تقديم قوالب التنبيهات"""
    template = alert_manager.template_env.get_template('model_performance_alert.html')
    
    html_content = template.render(
        model_id='test_model',
        metrics={'accuracy': 0.85},
        threshold=0.8,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    assert isinstance(html_content, str)
    assert 'test_model' in html_content
    assert '0.85' in html_content

def test_smtp_connection_error(mock_smtp):
    """اختبار خطأ اتصال SMTP"""
    mock_smtp.return_value.__enter__.side_effect = Exception("SMTP Error")
    
    with pytest.raises(Exception):
        alert_manager.send_alert(
            'Test Alert',
            'model_performance_alert',
            {},
            ['test@example.com']
        )

def test_invalid_template():
    """اختبار قالب غير صالح"""
    with pytest.raises(Exception):
        alert_manager.send_alert(
            'Test Alert',
            'invalid_template',
            {},
            ['test@example.com']
        ) 