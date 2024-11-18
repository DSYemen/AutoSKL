import pytest
import docker
import requests
import time
import subprocess
from pathlib import Path
import yaml
import json
from app.core.config import settings

@pytest.fixture
def docker_client():
    """عميل Docker للاختبار"""
    return docker.from_env()

@pytest.fixture
def app_container(docker_client):
    """حاوية التطبيق للاختبار"""
    # بناء الصورة
    image, _ = docker_client.images.build(
        path=".",
        tag="ml-framework:test"
    )
    
    # تشغيل الحاوية
    container = docker_client.containers.run(
        "ml-framework:test",
        detach=True,
        ports={'8000/tcp': 8000},
        environment={
            'DATABASE_URL': 'sqlite:///./test.db',
            'REDIS_URL': 'redis://redis:6379/0'
        }
    )
    
    # انتظار بدء التطبيق
    time.sleep(5)
    
    yield container
    
    # تنظيف
    container.stop()
    container.remove()

def test_docker_build():
    """اختبار بناء Docker"""
    result = subprocess.run(
        ['docker', 'build', '-t', 'ml-framework:test', '.'],
        capture_output=True
    )
    assert result.returncode == 0, "فشل بناء صورة Docker"

def test_container_health(app_container):
    """اختبار صحة الحاوية"""
    # التحقق من حالة الحاوية
    container_info = app_container.attrs
    assert container_info['State']['Status'] == 'running'
    assert container_info['State']['Health']['Status'] == 'healthy'

def test_environment_variables(app_container):
    """اختبار متغيرات البيئة"""
    # التحقق من تكوين المتغيرات
    env_vars = app_container.attrs['Config']['Env']
    assert any('DATABASE_URL' in var for var in env_vars)
    assert any('REDIS_URL' in var for var in env_vars)

def test_api_availability(app_container):
    """اختبار توفر API"""
    response = requests.get('http://localhost:8000/docs')
    assert response.status_code == 200
    assert 'swagger-ui' in response.text.lower()

def test_database_migration():
    """اختبار ترحيل قاعدة البيانات"""
    # تشغيل الترحيلات
    result = subprocess.run(
        ['alembic', 'upgrade', 'head'],
        capture_output=True
    )
    assert result.returncode == 0, "فشل ترحيل قاعدة البيانات"
    
    # التحقق من حالة الترحيل
    result = subprocess.run(
        ['alembic', 'current'],
        capture_output=True
    )
    assert 'head' in result.stdout.decode()

def test_static_files():
    """اختبار الملفات الثابتة"""
    static_dir = Path('static')
    assert static_dir.exists()
    
    # التحقق من وجود الملفات الأساسية
    assert (static_dir / 'css').exists()
    assert (static_dir / 'js').exists()
    assert (static_dir / 'images').exists()

def test_backup_restore():
    """اختبار النسخ الاحتياطي والاستعادة"""
    # إنشاء نسخة احتياطية
    result = subprocess.run(
        ['python', 'scripts/backup.py'],
        capture_output=True
    )
    assert result.returncode == 0
    
    # استعادة النسخة الاحتياطية
    backup_file = list(Path('backups').glob('*.dump'))[-1]
    result = subprocess.run(
        ['python', 'scripts/restore.py', str(backup_file)],
        capture_output=True
    )
    assert result.returncode == 0

def test_load_balancing():
    """اختبار توزيع الحمل"""
    # التحقق من تكوين Nginx
    nginx_conf = Path('nginx/nginx.conf')
    assert nginx_conf.exists()
    
    conf_text = nginx_conf.read_text()
    assert 'upstream' in conf_text
    assert 'proxy_pass' in conf_text

def test_ssl_configuration():
    """اختبار تكوين SSL"""
    nginx_conf = Path('nginx/nginx.conf')
    conf_text = nginx_conf.read_text()
    
    # التحقق من إعدادات SSL
    assert 'ssl_certificate' in conf_text
    assert 'ssl_protocols TLSv1.2 TLSv1.3' in conf_text

def test_monitoring_setup():
    """اختبار إعداد المراقبة"""
    # التحقق من تكوين Prometheus
    prometheus_conf = Path('monitoring/prometheus.yml')
    assert prometheus_conf.exists()
    
    # التحقق من تكوين Grafana
    grafana_dir = Path('monitoring/grafana')
    assert grafana_dir.exists()
    assert (grafana_dir / 'dashboards').exists()

def test_cache_configuration():
    """اختبار تكوين الذاكرة المؤقتة"""
    redis_conf = Path('redis/redis.conf')
    assert redis_conf.exists()
    
    conf_text = redis_conf.read_text()
    assert 'maxmemory' in conf_text
    assert 'maxmemory-policy' in conf_text

def test_logging_setup():
    """اختبار إعداد التسجيل"""
    log_dir = Path('logs')
    assert log_dir.exists()
    
    # التحقق من تدوير السجلات
    logrotate_conf = Path('logrotate.conf')
    assert logrotate_conf.exists()

def test_backup_schedule():
    """اختبار جدولة النسخ الاحتياطي"""
    crontab = Path('crontab')
    assert crontab.exists()
    
    crontab_text = crontab.read_text()
    assert 'backup.py' in crontab_text

def test_deployment_rollback():
    """اختبار التراجع عن النشر"""
    rollback_script = Path('scripts/rollback.py')
    assert rollback_script.exists()
    
    # محاكاة التراجع
    result = subprocess.run(
        ['python', str(rollback_script), '--dry-run'],
        capture_output=True
    )
    assert result.returncode == 0 