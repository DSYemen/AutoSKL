import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
from app.main import create_app
from app.db.database import get_db
from sqlalchemy.orm import Session
from app.ml.model_manager import model_manager
from app.ml.data_processing import data_processor
from app.ml.model_selection import model_selector
from app.utils.cache import cache_manager

client = TestClient(create_app())

@pytest.fixture
def sample_data():
    """بيانات تدريب للاختبار"""
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    return df

@pytest.fixture
async def trained_model(sample_data):
    """نموذج مدرب للاختبار"""
    # حفظ البيانات مؤقتاً
    temp_file = Path('temp_train.csv')
    sample_data.to_csv(temp_file, index=False)
    
    # تدريب النموذج
    with open(temp_file, 'rb') as f:
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", f, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
    
    temp_file.unlink()
    return response.json()["model_id"]

@pytest.mark.asyncio
async def test_complete_workflow(sample_data, trained_model):
    """اختبار سير العمل الكامل"""
    # 1. التحقق من تدريب النموذج
    model_info = client.get(f"/api/models/{trained_model}").json()
    assert model_info["task_type"] == "classification"
    
    # 2. إجراء تنبؤ
    prediction_data = {
        "data": {
            "feature1": 0.5,
            "feature2": 0.5
        },
        "return_probabilities": True
    }
    
    prediction = client.post(
        f"/api/predict/{trained_model}",
        json=prediction_data
    ).json()
    
    assert "predictions" in prediction
    assert len(prediction["predictions"]) == 1
    
    # 3. تقييم النموذج
    # حفظ بيانات التقييم
    temp_eval = Path('temp_eval.csv')
    sample_data.to_csv(temp_eval, index=False)
    
    with open(temp_eval, 'rb') as f:
        evaluation = client.post(
            f"/api/models/{trained_model}/evaluate",
            files={"file": ("eval.csv", f, "text/csv")},
            data={"target_column": "target"}
        ).json()
    
    temp_eval.unlink()
    assert "evaluation_results" in evaluation

@pytest.mark.asyncio
async def test_data_processing_integration():
    """اختبار تكامل معالجة البيانات"""
    df = pd.DataFrame({
        'numeric': [1.0, 2.0, np.nan, 4.0],
        'categorical': ['A', 'B', None, 'A'],
        'target': [0, 1, 0, 1]
    })
    
    X, y = data_processor.process_data(df, 'target', 'classification')
    assert not np.isnan(X).any()  # التحقق من معالجة القيم المفقودة
    assert X.shape[1] > df.shape[1] - 1  # التحقق من الترميز One-hot

@pytest.mark.asyncio
async def test_model_selection_integration():
    """اختبار تكامل اختيار النموذج"""
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    
    model, params = model_selector.select_best_model(
        X, y,
        'classification',
        n_trials=5  # عدد قليل للاختبار
    )
    
    assert hasattr(model, 'predict')
    assert isinstance(params, dict)

@pytest.mark.asyncio
async def test_caching_integration():
    """اختبار تكامل التخزين المؤقت"""
    key = 'test_key'
    data = {'test': 'data'}
    
    # تخزين البيانات
    await cache_manager.set(key, data)
    
    # استرجاع البيانات
    cached_data = await cache_manager.get(key)
    assert cached_data == data
    
    # حذف البيانات
    await cache_manager.delete(key)
    assert await cache_manager.get(key) is None

@pytest.mark.asyncio
async def test_database_integration():
    """اختبار تكامل قاعدة البيانات"""
    async for db in get_db():
        # إنشاء نموذج جديد
        model_data = {
            'model_id': 'test_integration',
            'task_type': 'classification',
            'target_column': 'target',
            'metadata': {'test': 'data'}
        }
        
        # حفظ في قاعدة البيانات
        db_model = Model(**model_data)
        db.add(db_model)
        await db.commit()
        
        # استرجاع من قاعدة البيانات
        result = await db.query(Model).filter(
            Model.model_id == 'test_integration'
        ).first()
        
        assert result.task_type == 'classification'
        
        # تنظيف
        await db.delete(result)
        await db.commit()

@pytest.mark.asyncio
async def test_monitoring_integration(trained_model):
    """اختبار تكامل المراقبة"""
    async for db in get_db():
        # إضافة مقاييس أداء
        metrics = {
            'accuracy': 0.85,
            'precision': 0.83
        }
        
        result = await model_monitor.check_model_performance(
            trained_model,
            metrics,
            db
        )
        
        assert isinstance(result, bool)
        
        # التحقق من انحراف البيانات
        summary = await model_monitor.get_monitoring_summary(
            trained_model,
            db
        )
        
        assert 'metrics_history' in summary
        assert 'drift_records' in summary

@pytest.mark.asyncio
async def test_error_handling_integration():
    """اختبار تكامل معالجة الأخطاء"""
    # محاولة الوصول إلى نموذج غير موجود
    response = client.get("/api/models/nonexistent_model")
    assert response.status_code == 404
    
    # محاولة تدريب نموذج بدون بيانات
    response = client.post(
        "/api/train",
        files={},
        data={
            "task_type": "classification",
            "target_column": "target"
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_cleanup_integration():
    """اختبار تكامل التنظيف"""
    # تنظيف الذاكرة المؤقتة
    await cache_manager.clear()
    
    # تنظيف النماذج
    models = model_manager.list_models()
    for model_id in models:
        model_manager.delete_model(model_id)
    
    # التحقق من التنظيف
    assert len(model_manager.list_models()) == 0 