import pytest
from fastapi.testclient import TestClient
import yaml
from app.main import create_app
from app.utils.documentation import documentation_generator

client = TestClient(create_app())

def test_openapi_schema():
    """اختبار مخطط OpenAPI"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    # التحقق من المعلومات الأساسية
    assert schema["info"]["title"] == "AutoML Framework"
    assert "version" in schema["info"]

def test_api_documentation_generation():
    """اختبار توليد توثيق API"""
    docs = documentation_generator.generate_api_documentation(create_app())
    
    assert isinstance(docs, dict)
    assert "openapi" in docs
    assert "paths" in docs
    
    # التحقق من توثيق المسارات الرئيسية
    paths = docs["paths"]
    assert "/api/train" in paths
    assert "/api/predict/{model_id}" in paths
    assert "/api/models" in paths

def test_swagger_ui():
    """اختبار واجهة Swagger"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()

def test_redoc():
    """اختبار واجهة ReDoc"""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()

def test_endpoint_documentation():
    """اختبار توثيق نقاط النهاية"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    # التحقق من توثيق نقطة نهاية التدريب
    train_endpoint = schema["paths"]["/api/train"]["post"]
    assert "summary" in train_endpoint
    assert "description" in train_endpoint
    assert "requestBody" in train_endpoint
    assert "responses" in train_endpoint

def test_model_schemas():
    """اختبار مخططات النماذج"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    assert "components" in schema
    assert "schemas" in schema["components"]
    
    schemas = schema["components"]["schemas"]
    assert "PredictionRequest" in schemas
    assert "PredictionResponse" in schemas
    assert "ModelInfo" in schemas

def test_api_examples():
    """اختبار أمثلة API"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    # التحقق من وجود أمثلة في المخططات
    for path in schema["paths"].values():
        for operation in path.values():
            if "requestBody" in operation:
                assert "content" in operation["requestBody"]
                content = operation["requestBody"]["content"]
                assert any("example" in v or "examples" in v for v in content.values())

def test_response_codes():
    """اختبار رموز الاستجابة"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    for path in schema["paths"].values():
        for operation in path.values():
            assert "responses" in operation
            responses = operation["responses"]
            assert "200" in responses or "201" in responses
            assert "400" in responses or "422" in responses

def test_parameter_descriptions():
    """اختبار وصف المعلمات"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    for path in schema["paths"].values():
        for operation in path.values():
            if "parameters" in operation:
                for param in operation["parameters"]:
                    assert "description" in param
                    assert param["description"].strip() != ""

def test_markdown_descriptions():
    """اختبار وصف Markdown"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    def check_markdown(text):
        if not text:
            return True
        # التحقق من تنسيق Markdown الأساسي
        return any(marker in text for marker in ['#', '*', '_', '`', '-'])
    
    for path in schema["paths"].values():
        for operation in path.values():
            if "description" in operation:
                assert check_markdown(operation["description"])

def test_api_tags():
    """اختبار وسوم API"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    assert "tags" in schema
    tags = schema["tags"]
    
    # التحقق من الوسوم الأساسية
    tag_names = [tag["name"] for tag in tags]
    assert "Models" in tag_names
    assert "Predictions" in tag_names
    assert "Training" in tag_names

def test_security_schemes():
    """اختبار مخططات الأمان"""
    response = client.get("/openapi.json")
    schema = response.json()
    
    assert "components" in schema
    assert "securitySchemes" in schema["components"]
    security_schemes = schema["components"]["securitySchemes"]
    
    # التحقق من تكوين الأمان
    assert any(scheme["type"] == "http" for scheme in security_schemes.values())

def test_yaml_export():
    """اختبار تصدير YAML"""
    docs = documentation_generator.generate_api_documentation(create_app())
    
    # التحقق من إمكانية تصدير التوثيق إلى YAML
    yaml_docs = yaml.dump(docs, allow_unicode=True)
    assert isinstance(yaml_docs, str)
    assert len(yaml_docs) > 0
    
    # التحقق من إمكانية تحميل YAML
    loaded_docs = yaml.safe_load(yaml_docs)
    assert loaded_docs == docs 