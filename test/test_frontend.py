import pytest
from fastapi.testclient import TestClient
from bs4 import BeautifulSoup
from app.main import create_app
import json
from pathlib import Path

client = TestClient(create_app())

@pytest.fixture
def sample_models():
    """نماذج نموذجية للاختبار"""
    return [
        {
            'id': 'model1',
            'type': 'classification',
            'accuracy': 0.85,
            'updated_at': '2024-01-01 12:00:00'
        },
        {
            'id': 'model2',
            'type': 'regression',
            'accuracy': 0.78,
            'updated_at': '2024-01-02 12:00:00'
        }
    ]

def test_index_page():
    """اختبار صفحة الرئيسية"""
    response = client.get("/")
    assert response.status_code == 200
    
    # تحليل HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # التحقق من العناصر الرئيسية
    assert soup.find('title').text == 'إطار عمل التعلم الآلي'
    assert soup.find('div', class_='model-stats')
    assert soup.find('div', class_='upload-area')

def test_model_stats_section(sample_models):
    """اختبار قسم إحصائيات النماذج"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    stats = soup.find('div', class_='model-stats')
    stats_items = stats.find_all('div', class_='stats-item')
    
    assert len(stats_items) == 4  # إجمالي النماذج، النماذج النشطة، التنبؤات، متوسط الدقة
    assert 'إجمالي النماذج' in stats.text
    assert 'النماذج النشطة' in stats.text

def test_upload_form():
    """اختبار نموذج الرفع"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    form = soup.find('form')
    assert form is not None
    
    # التحقق من حقول النموذج
    assert form.find('select', {'name': 'task_type'})
    assert form.find('input', {'name': 'target_column'})
    assert form.find('input', {'id': 'fileInput'})

def test_recent_models_table(sample_models):
    """اختبار جدول النماذج الحديثة"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table')
    assert table is not None
    
    headers = [th.text.strip() for th in table.find_all('th')]
    assert 'معرف النموذج' in headers
    assert 'النوع' in headers
    assert 'الدقة' in headers
    assert 'التحديث' in headers

def test_file_upload():
    """اختبار رفع الملف"""
    # إنشاء ملف CSV للاختبار
    test_file = Path('test.csv')
    test_file.write_text('feature1,feature2,target\n1,2,0\n3,4,1')
    
    with open(test_file, 'rb') as f:
        response = client.post(
            "/api/train",
            files={"file": ("test.csv", f, "text/csv")},
            data={
                "task_type": "classification",
                "target_column": "target"
            }
        )
    
    test_file.unlink()  # تنظيف
    assert response.status_code == 200

def test_model_actions():
    """اختبار إجراءات النموذج"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    action_buttons = soup.find_all('button', class_='btn-sm')
    assert len(action_buttons) >= 2  # زر العرض وزر الحذف لكل نموذج
    
    # التحقق من وجود أيقونات الإجراءات
    assert soup.find('i', class_='bi-eye')  # أيقونة العرض
    assert soup.find('i', class_='bi-trash')  # أيقونة الحذف

def test_responsive_design():
    """اختبار التصميم المتجاوب"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # التحقق من وجود عناصر التصميم المتجاوب
    assert soup.find('meta', {'name': 'viewport'})
    assert soup.find('div', class_='container-fluid')
    assert soup.find('div', class_='row')
    assert soup.find('div', class_='col-md-6')

def test_drag_drop_functionality():
    """اختبار وظيفة السحب والإفلات"""
    response = client.get("/")
    soup = BeautifulSoup(response.text, 'html.parser')
    
    upload_area = soup.find('div', class_='upload-area')
    assert upload_area is not None
    assert 'اسحب ملف البيانات هنا' in upload_area.text

def test_error_handling():
    """اختبار معالجة الأخطاء"""
    # محاولة رفع ملف غير صالح
    response = client.post(
        "/api/train",
        files={"file": ("test.txt", b"invalid data", "text/plain")},
        data={
            "task_type": "classification",
            "target_column": "target"
        }
    )
    assert response.status_code == 400

def test_javascript_functionality():
    """اختبار وظائف JavaScript"""
    response = client.get("/")
    
    # التحقق من تحميل Bootstrap
    assert 'bootstrap.bundle.min.js' in response.text
    
    # التحقق من وجود معالجات الأحداث
    assert 'addEventListener' in response.text
    assert 'handleFile' in response.text 