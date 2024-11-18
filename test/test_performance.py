import pytest
import time
import asyncio
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from app.main import create_app
from app.ml.model_manager import model_manager
from app.utils.cache import cache_manager

client = TestClient(create_app())

@pytest.fixture
def trained_model():
    """نموذج مدرب للاختبار"""
    # إنشاء بيانات تدريب
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # تدريب النموذج
    response = client.post(
        "/api/train",
        files={"file": ("data.csv", df.to_csv(index=False))},
        data={
            "task_type": "classification",
            "target_column": "target"
        }
    )
    
    return response.json()["model_id"]

def test_model_training_performance():
    """اختبار أداء تدريب النموذج"""
    # إنشاء مجموعات بيانات مختلفة الأحجام
    sizes = [100, 1000, 10000]
    training_times = []
    
    for size in sizes:
        df = pd.DataFrame({
            'feature1': np.random.rand(size),
            'feature2': np.random.rand(size),
            'target': np.random.randint(0, 2, size)
        })
        
        start_time = time.time()
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", df.to_csv(index=False))},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        assert response.status_code == 200
        
    # التحقق من أن وقت التدريب يزداد بشكل معقول مع حجم البيانات
    assert all(t2 > t1 for t1, t2 in zip(training_times[:-1], training_times[1:]))

@pytest.mark.asyncio
async def test_prediction_latency(trained_model):
    """اختبار زمن الاستجابة للتنبؤات"""
    prediction_data = {
        "data": {
            "feature1": 0.5,
            "feature2": 0.5
        },
        "return_probabilities": True
    }
    
    latencies = []
    for _ in range(100):  # 100 طلب
        start_time = time.time()
        response = client.post(
            f"/api/predict/{trained_model}",
            json=prediction_data
        )
        latency = time.time() - start_time
        latencies.append(latency)
        
        assert response.status_code == 200
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    # التحقق من أن متوسط زمن الاستجابة أقل من 100 مللي ثانية
    assert avg_latency < 0.1
    # التحقق من أن 95% من الطلبات تستجيب في أقل من 200 مللي ثانية
    assert p95_latency < 0.2

def test_concurrent_predictions(trained_model):
    """اختبار التنبؤات المتزامنة"""
    n_requests = 50  # عدد الطلبات المتزامنة
    
    def make_prediction():
        return client.post(
            f"/api/predict/{trained_model}",
            json={
                "data": {
                    "feature1": 0.5,
                    "feature2": 0.5
                }
            }
        )
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        responses = list(executor.map(lambda _: make_prediction(), range(n_requests)))
    
    # التحقق من نجاح جميع الطلبات
    assert all(r.status_code == 200 for r in responses)

@pytest.mark.asyncio
async def test_cache_performance():
    """اختبار أداء التخزين المؤقت"""
    key = 'perf_test'
    data = {'large_array': np.random.rand(1000, 1000).tolist()}
    
    # قياس وقت التخزين
    start_time = time.time()
    await cache_manager.set(key, data)
    write_time = time.time() - start_time
    
    # قياس وقت القراءة
    start_time = time.time()
    cached_data = await cache_manager.get(key)
    read_time = time.time() - start_time
    
    # التحقق من أن عمليات التخزين المؤقت سريعة
    assert write_time < 0.1  # أقل من 100 مللي ثانية
    assert read_time < 0.01  # أقل من 10 مللي ثانية

@pytest.mark.asyncio
async def test_model_memory_usage():
    """اختبار استخدام الذاكرة"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # تدريب عدة نماذج
    for _ in range(5):
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
        
        assert response.status_code == 200
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # التحقق من أن الزيادة في استخدام الذاكرة معقولة
    assert memory_increase < 500 * 1024 * 1024  # أقل من 500 ميجابايت

def test_database_performance():
    """اختبار أداء قاعدة البيانات"""
    from app.db.database import get_db
    import time
    
    async def measure_db_operations():
        async for db in get_db():
            # قياس وقت الكتابة
            start_time = time.time()
            for i in range(100):
                model = Model(
                    model_id=f'perf_test_{i}',
                    task_type='classification',
                    target_column='target'
                )
                db.add(model)
            await db.commit()
            write_time = time.time() - start_time
            
            # قياس وقت القراءة
            start_time = time.time()
            models = await db.query(Model).all()
            read_time = time.time() - start_time
            
            # تنظيف
            for model in models:
                await db.delete(model)
            await db.commit()
            
            return write_time, read_time
    
    write_time, read_time = asyncio.run(measure_db_operations())
    
    # التحقق من أن عمليات قاعدة البيانات سريعة
    assert write_time < 1.0  # أقل من ثانية
    assert read_time < 0.1  # أقل من 100 مللي ثانية

@pytest.mark.asyncio
async def test_api_response_size():
    """اختبار حجم استجابات API"""
    # الحصول على قائمة النماذج
    response = client.get("/api/models")
    response_size = len(response.content)
    
    # التحقق من أن حجم الاستجابة معقول
    assert response_size < 1024 * 1024  # أقل من 1 ميجابايت 