import pytest
import asyncio
import signal
import psutil
import time
from pathlib import Path
import subprocess
from fastapi.testclient import TestClient
from app.main import create_app
from app.db.database import engine, Base
from app.utils.cache import cache_manager
from app.core.config import settings

client = TestClient(create_app())

@pytest.fixture
async def setup_test_environment():
    """إعداد بيئة الاختبار"""
    # إنشاء قاعدة بيانات اختبار
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # تنظيف
    await cache_manager.clear()

async def test_database_connection_recovery():
    """اختبار استرداد اتصال قاعدة البيانات"""
    # محاكاة فقدان الاتصال
    await engine.dispose()
    
    # محاولة إجراء عملية
    response = client.get("/api/models")
    
    # يجب أن يعيد الاتصال تلقائياً
    assert response.status_code != 500
    assert engine.pool.checkedout() == 0

async def test_cache_recovery():
    """اختبار استرداد الذاكرة المؤقتة"""
    # محاكاة فشل Redis
    original_redis = cache_manager.redis_client
    cache_manager.redis_client = None
    
    try:
        # محاولة استخدام الذاكرة المؤقتة
        key = 'test_key'
        value = {'test': 'data'}
        
        # يجب أن يستخدم التخزين المؤقت المحلي
        await cache_manager.set(key, value)
        cached_value = await cache_manager.get(key)
        assert cached_value == value
        
    finally:
        cache_manager.redis_client = original_redis

async def test_process_recovery():
    """اختبار استرداد العمليات"""
    def kill_worker():
        for proc in psutil.process_iter(['pid', 'name']):
            if 'celery' in proc.info['name']:
                proc.kill()
                break
    
    # قتل عملية Celery
    kill_worker()
    
    # انتظار إعادة التشغيل التلقائي
    time.sleep(5)
    
    # التحقق من تشغيل العملية
    celery_running = any('celery' in proc.name() 
                        for proc in psutil.process_iter(['name']))
    assert celery_running

async def test_file_system_recovery():
    """اختبار استرداد نظام الملفات"""
    # محاكاة فشل الكتابة
    test_dir = Path('test_recovery')
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / 'test.txt'
    
    try:
        # محاولة الكتابة في مجلد محمي
        test_file.chmod(0o000)
        
        # يجب أن يستخدم مجلد بديل
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", b"test,data\n1,2")},
            data={"task_type": "classification"}
        )
        assert response.status_code != 500
        
    finally:
        test_file.chmod(0o666)
        test_dir.rmdir()

async def test_memory_recovery():
    """اختبار استرداد الذاكرة"""
    # محاكاة استهلاك الذاكرة
    large_data = [0] * (1024 * 1024 * 100)  # 100MB
    
    try:
        # محاولة إجراء عملية
        response = client.get("/api/models")
        assert response.status_code == 200
        
    finally:
        del large_data

async def test_network_recovery():
    """اختبار استرداد الشبكة"""
    # محاكاة مشاكل الشبكة
    async def simulate_network_issues():
        subprocess.run(['tc', 'qdisc', 'add', 'dev', 'lo', 'root', 'netem', 'loss', '50%'])
        await asyncio.sleep(5)
        subprocess.run(['tc', 'qdisc', 'del', 'dev', 'lo', 'root'])
    
    asyncio.create_task(simulate_network_issues())
    
    # محاولة إجراء عدة طلبات
    success = 0
    for _ in range(10):
        try:
            response = client.get("/api/models")
            if response.status_code == 200:
                success += 1
        except:
            pass
        await asyncio.sleep(1)
    
    assert success > 0

async def test_deadlock_recovery():
    """اختبار استرداد التوقف المتبادل"""
    async def simulate_deadlock():
        # محاكاة توقف متبادل في قاعدة البيانات
        async with engine.begin() as conn:
            await conn.execute("SELECT pg_advisory_lock(1)")
            await conn.execute("SELECT pg_advisory_lock(2)")
    
    # يجب أن ينتهي بمهلة زمنية
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(simulate_deadlock(), timeout=5)

async def test_crash_recovery():
    """اختبار استرداد الانهيار"""
    def simulate_crash():
        os.kill(os.getpid(), signal.SIGSEGV)
    
    # تسجيل معالج الإشارة
    original_handler = signal.getsignal(signal.SIGSEGV)
    signal.signal(signal.SIGSEGV, lambda sig, frame: None)
    
    try:
        simulate_crash()
    except:
        pass
    finally:
        signal.signal(signal.SIGSEGV, original_handler)
    
    # التحقق من استمرار عمل التطبيق
    response = client.get("/")
    assert response.status_code == 200

async def test_data_corruption_recovery():
    """اختبار استرداد تلف البيانات"""
    # محاكاة تلف البيانات
    model_file = Path('models/test_model.joblib')
    if model_file.exists():
        original_content = model_file.read_bytes()
        try:
            # كتابة بيانات تالفة
            model_file.write_bytes(b'corrupted data')
            
            # محاولة تحميل النموذج
            response = client.get("/api/models/test_model")
            
            # يجب أن يستخدم النسخة الاحتياطية
            assert response.status_code != 500
            
        finally:
            if original_content:
                model_file.write_bytes(original_content)

async def test_resource_exhaustion_recovery():
    """اختبار استرداد استنفاد الموارد"""
    # محاكاة استنفاد الموارد
    processes = []
    try:
        # إنشاء عمليات كثيرة
        for _ in range(100):
            process = subprocess.Popen(['sleep', '1'])
            processes.append(process)
        
        # محاولة إجراء عملية
        response = client.get("/api/models")
        assert response.status_code == 200
        
    finally:
        for process in processes:
            process.kill()

async def test_state_recovery():
    """اختبار استرداد الحالة"""
    # محاكاة فقدان الحالة
    original_state = cache_manager.get_state()
    try:
        await cache_manager.clear_state()
        
        # محاولة إجراء عملية
        response = client.get("/api/models")
        
        # يجب أن يعيد بناء الحالة
        assert response.status_code == 200
        assert await cache_manager.get_state() is not None
        
    finally:
        await cache_manager.set_state(original_state) 