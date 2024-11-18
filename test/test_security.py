# import pytest
# from fastapi.testclient import TestClient
# import jwt
# from datetime import datetime, timedelta
# from app.main import create_app
# # from app.core.config import settings
# import json
# import base64

# client = TestClient(create_app())

# @pytest.fixture
# def valid_token():
#     """توليد رمز JWT صالح"""
#     payload = {
#         'sub': 'test_user',
#         'exp': datetime.utcnow() + timedelta(minutes=30)
#     }
#     return jwt.encode(payload, settings.secret_key, algorithm='HS256')

# def test_unauthorized_access():
#     """اختبار الوصول غير المصرح به"""
#     # محاولة الوصول بدون مصادقة
#     response = client.get("/api/models")
#     assert response.status_code == 401

# def test_invalid_token():
#     """اختبار رمز غير صالح"""
#     headers = {'Authorization': 'Bearer invalid_token'}
#     response = client.get("/api/models", headers=headers)
#     assert response.status_code == 401

# def test_expired_token():
#     """اختبار رمز منتهي الصلاحية"""
#     payload = {
#         'sub': 'test_user',
#         'exp': datetime.utcnow() - timedelta(minutes=30)
#     }
#     token = jwt.encode(payload, settings.secret_key, algorithm='HS256')
    
#     headers = {'Authorization': f'Bearer {token}'}
#     response = client.get("/api/models", headers=headers)
#     assert response.status_code == 401

# def test_sql_injection():
#     """اختبار حقن SQL"""
#     malicious_input = "'; DROP TABLE models; --"
    
#     response = client.get(f"/api/models/{malicious_input}")
#     assert response.status_code in [400, 404]  # يجب رفض المدخلات الخبيثة

# def test_xss_prevention():
#     """اختبار منع XSS"""
#     malicious_script = "<script>alert('xss')</script>"
    
#     # محاولة إرسال نص برمجي خبيث في البيانات
#     response = client.post(
#         "/api/train",
#         json={
#             "model_name": malicious_script,
#             "task_type": "classification"
#         }
#     )
    
#     assert malicious_script not in response.text

# def test_file_upload_security():
#     """اختبار أمان رفع الملفات"""
#     # محاولة رفع ملف تنفيذي
#     malicious_file = b"MZ\x90\x00\x03\x00\x00\x00"  # رأس PE
    
#     response = client.post(
#         "/api/train",
#         files={"file": ("malicious.exe", malicious_file, "application/x-msdownload")},
#         data={"task_type": "classification"}
#     )
    
#     assert response.status_code == 400

# def test_rate_limiting():
#     """اختبار تحديد معدل الطلبات"""
#     # إرسال العديد من الطلبات بسرعة
#     responses = []
#     for _ in range(100):
#         response = client.get("/api/models")
#         responses.append(response)
    
#     # يجب أن يتم تقييد بعض الطلبات
#     assert any(r.status_code == 429 for r in responses)

# def test_sensitive_data_exposure():
#     """اختبار تسرب البيانات الحساسة"""
#     response = client.get("/api/models")
    
#     # التحقق من عدم تسرب معلومات حساسة
#     assert 'password' not in response.text
#     assert 'secret' not in response.text
#     assert 'token' not in response.text

# def test_cors_policy():
#     """اختبار سياسة CORS"""
#     headers = {'Origin': 'http://malicious-site.com'}
#     response = client.get("/api/models", headers=headers)
    
#     assert 'Access-Control-Allow-Origin' not in response.headers

# def test_content_security_policy():
#     """اختبار سياسة أمان المحتوى"""
#     response = client.get("/")
    
#     assert 'Content-Security-Policy' in response.headers
#     csp = response.headers['Content-Security-Policy']
#     assert "default-src 'self'" in csp

# def test_secure_headers():
#     """اختبار الترويسات الآمنة"""
#     response = client.get("/")
    
#     assert 'X-Content-Type-Options' in response.headers
#     assert 'X-Frame-Options' in response.headers
#     assert 'X-XSS-Protection' in response.headers

# def test_password_hashing():
#     """اختبار تشفير كلمات المرور"""
#     from app.utils.security import hash_password
    
#     password = "test_password"
#     hashed = hash_password(password)
    
#     assert password != hashed
#     assert len(hashed) > 50  # التحقق من أن التشفير قوي

# def test_input_validation():
#     """اختبار التحقق من صحة المدخلات"""
#     invalid_inputs = [
#         {'task_type': 'invalid_type'},
#         {'task_type': None},
#         {'task_type': '; DROP TABLE models; --'}
#     ]
    
#     for invalid_input in invalid_inputs:
#         response = client.post("/api/train", json=invalid_input)
#         assert response.status_code in [400, 422]

# def test_error_messages():
#     """اختبار رسائل الخطأ"""
#     response = client.get("/api/models/nonexistent")
    
#     # التحقق من أن رسائل الخطأ لا تكشف معلومات حساسة
#     assert 'stack trace' not in response.text
#     assert 'database' not in response.text.lower() 