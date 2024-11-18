import pytest
import time
import asyncio
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from locust import HttpUser, task, between
from fastapi.testclient import TestClient
from app.main import create_app
from app.ml.model_manager import model_manager
from app.utils.cache import cache_manager
import psutil
import os
import gc
import cProfile
import pstats
import memory_profiler

client = TestClient(create_app())

class MLFrameworkUser(HttpUser):
    """مستخدم Locust لاختبار الأداء"""
    wait_time = between(1, 3)
    
    @task(3)
    def predict(self):
        """اختبار التنبؤ"""
        self.client.post(
            f"/api/predict/{self.model_id}",
            json={
                "data": {
                    "feature1": np.random.rand(),
                    "feature2": np.random.rand()
                }
            }
        )
    
    @task(1)
    def get_model_info(self):
        """اختبار معلومات النموذج"""
        self.client.get(f"/api/models/{self.model_id}")

@pytest.fixture
def profiled_model():
    """نموذج مع تتبع الأداء"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # تدريب النموذج
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    response = client.post(
        "/api/train",
        files={"file": ("data.csv", df.to_csv(index=False))},
        data={
            "task_type": "classification",
            "target_column": "target"
        }
    )
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    
    return response.json()["model_id"], stats

@memory_profiler.profile
def test_memory_leak():
    """اختبار تسرب الذاكرة"""
    initial_memory = psutil.Process().memory_info().rss
    
    for _ in range(100):
        # عمليات متكررة
        df = pd.DataFrame(np.random.rand(100, 10))
        model_manager.get_models()
        gc.collect()
    
    final_memory = psutil.Process().memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # يجب أن تكون الزيادة في الذاكرة ضئيلة
    assert memory_increase < 10 * 1024 * 1024  # أقل من 10 ميجابايت

async def test_async_performance():
    """اختبار أداء العمليات المتزامنة"""
    async def make_request():
        start = time.time()
        await asyncio.sleep(0.1)  # محاكاة طلب شبكة
        return time.time() - start
    
    # تنفيذ عدة طلبات متزامنة
    tasks = [make_request() for _ in range(100)]
    durations = await asyncio.gather(*tasks)
    
    # التحقق من أن الطلبات المتزامنة أسرع من التسلسلية
    total_time = sum(durations)
    sequential_time = 0.1 * 100
    assert total_time < sequential_time

def test_cpu_intensive_operations():
    """اختبار العمليات كثيفة المعالجة"""
    def heavy_computation():
        # محاكاة عملية ثقيلة
        matrix = np.random.rand(1000, 1000)
        return np.linalg.svd(matrix)
    
    with ProcessPoolExecutor() as executor:
        start = time.time()
        list(executor.map(heavy_computation, range(4)))
        parallel_time = time.time() - start
        
        start = time.time()
        for _ in range(4):
            heavy_computation()
        sequential_time = time.time() - start
        
        assert parallel_time < sequential_time

def test_cache_hit_ratio():
    """اختبار نسبة إصابة الذاكرة المؤقتة"""
    hits = 0
    total = 1000
    
    for i in range(total):
        key = f'test_key_{i % 10}'  # استخدام 10 مفاتيح فقط
        if cache_manager.exists(key):
            hits += 1
        else:
            cache_manager.set(key, f'value_{i}')
    
    hit_ratio = hits / total
    assert hit_ratio > 0.8  # يجب أن تكون نسبة الإصابة أعلى من 80%

def test_database_connection_pool():
    """اختبار مجمع اتصالات قاعدة البيانات"""
    from app.db.database import engine
    
    # التحقق من إعدادات المجمع
    pool = engine.pool
    assert pool.size() >= 5  # الحد الأدنى لحجم المجمع
    assert pool.overflow() == -1  # لا حد أقصى للاتصالات الإضافية

def test_response_compression():
    """اختبار ضغط الاستجابات"""
    headers = {'Accept-Encoding': 'gzip'}
    response = client.get("/api/models", headers=headers)
    
    assert response.headers.get('Content-Encoding') == 'gzip'
    assert len(response.content) < 1024 * 1024  # أقل من 1 ميجابايت

def test_batch_processing():
    """اختبار معالجة الدفعات"""
    batch_size = 1000
    data = [
        {
            'feature1': np.random.rand(),
            'feature2': np.random.rand()
        }
        for _ in range(batch_size)
    ]
    
    start = time.time()
    response = client.post(
        "/api/predict_batch",
        json={"data": data}
    )
    batch_time = time.time() - start
    
    # مقارنة مع المعالجة الفردية
    start = time.time()
    for item in data:
        client.post(
            "/api/predict",
            json={"data": item}
        )
    individual_time = time.time() - start
    
    assert batch_time < individual_time / 2

def test_load_balancing():
    """اختبار توزيع الحمل"""
    import multiprocessing
    
    def worker():
        return client.get("/api/models").status_code
    
    # تشغيل عدة عمليات متوازية
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(worker, range(100)))
    
    # التحقق من نجاح جميع الطلبات
    assert all(code == 200 for code in results)

def test_resource_cleanup():
    """اختبار تنظيف الموارد"""
    initial_handles = psutil.Process().num_handles()
    
    for _ in range(100):
        client.get("/api/models")
        gc.collect()
    
    final_handles = psutil.Process().num_handles()
    assert final_handles <= initial_handles + 10  # سماح بزيادة صغيرة 