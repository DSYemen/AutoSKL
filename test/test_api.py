import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from pathlib import Path
import json
from app.main import create_app
from app.ml.model_manager import model_manager
from app.db.database import get_db

app = create_app()
client = TestClient(app)

@pytest.fixture
def sample_data():
    """بيانات اختبار"""
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

@pytest.fixture
def trained_model_id(sample_data):
    """نموذج مدرب للاختبار"""
    # حفظ البيانات مؤقتاً
    temp_file = Path("temp_data.csv")
    sample_data.to_csv(temp_file, index=False)
    
    # تدريب النموذج
    with open(temp_file, "rb") as f:
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", f, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
    
    # تنظيف
    temp_file.unlink()
    
    assert response.status_code == 200
    return response.json()["model_id"]

def test_train_model(sample_data):
    """اختبار تدريب النموذج"""
    # حفظ البيانات مؤقتاً
    temp_file = Path("temp_data.csv")
    sample_data.to_csv(temp_file, index=False)
    
    # إرسال طلب التدريب
    with open(temp_file, "rb") as f:
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", f, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
    
    # تنظيف
    temp_file.unlink()
    
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "task_type" in data
    assert "evaluation_results" in data

def test_predict(trained_model_id):
    """اختبار التنبؤ"""
    # بيانات التنبؤ
    prediction_data = {
        "data": {
            "feature1": 0.5,
            "feature2": 0.7
        },
        "return_probabilities": True
    }
    
    response = client.post(
        f"/api/predict/{trained_model_id}",
        json=prediction_data
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert "prediction" in data["predictions"][0]

def test_list_models(trained_model_id):
    """اختبار قائمة النماذج"""
    response = client.get("/api/models")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert any(model["model_id"] == trained_model_id for model in data)

def test_get_model_info(trained_model_id):
    """اختبار معلومات النموذج"""
    response = client.get(f"/api/models/{trained_model_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == trained_model_id
    assert "task_type" in data
    assert "evaluation_results" in data

def test_delete_model(trained_model_id):
    """اختبار حذف النموذج"""
    response = client.delete(f"/api/models/{trained_model_id}")
    
    assert response.status_code == 200
    assert response.json()["message"].startswith("تم حذف النموذج")
    
    # التأكد من عدم وجود النموذج
    response = client.get(f"/api/models/{trained_model_id}")
    assert response.status_code == 404

def test_evaluate_model(trained_model_id, sample_data):
    """اختبار تقييم النموذج"""
    # حفظ بيانات التقييم مؤقتاً
    temp_file = Path("temp_eval_data.csv")
    sample_data.to_csv(temp_file, index=False)
    
    # إرسال طلب التقييم
    with open(temp_file, "rb") as f:
        response = client.post(
            f"/api/models/{trained_model_id}/evaluate",
            files={"file": ("eval_data.csv", f, "text/csv")},
            data={"target_column": "target"}
        )
    
    # تنظيف
    temp_file.unlink()
    
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "evaluation_results" in data

def test_invalid_model_id():
    """اختبار معرف نموذج غير صالح"""
    response = client.get("/api/models/invalid_id")
    assert response.status_code == 404

def test_invalid_file_format():
    """اختبار تنسيق ملف غير صالح"""
    response = client.post(
        "/api/train",
        files={"file": ("data.txt", b"invalid data", "text/plain")},
        data={
            "task_type": "classification",
            "target_column": "target"
        }
    )
    assert response.status_code == 400

def test_missing_target_column(sample_data):
    """اختبار عمود هدف مفقود"""
    # حذف عمود الهدف
    data_without_target = sample_data.drop('target', axis=1)
    
    # حفظ البيانات مؤقتاً
    temp_file = Path("temp_data.csv")
    data_without_target.to_csv(temp_file, index=False)
    
    # إرسال طلب التدريب
    with open(temp_file, "rb") as f:
        response = client.post(
            "/api/train",
            files={"file": ("data.csv", f, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
    
    # تنظيف
    temp_file.unlink()
    
    assert response.status_code == 422 