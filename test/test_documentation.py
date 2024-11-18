import pytest
from fastapi import FastAPI
from pathlib import Path
import yaml
from app.utils.documentation import documentation_generator

@pytest.fixture
def sample_app():
    """تطبيق FastAPI للاختبار"""
    app = FastAPI()
    
    @app.get("/api/test")
    def test_endpoint():
        """نقطة نهاية اختبار"""
        return {"message": "test"}
        
    @app.post("/api/model")
    def model_endpoint():
        """نقطة نهاية نموذج"""
        return {"status": "success"}
        
    return app

@pytest.fixture
def sample_model_info():
    """معلومات نموذج للاختبار"""
    return {
        'task_type': 'classification',
        'target_column': 'target',
        'preprocessing_info': {
            'numeric_features': ['feature1', 'feature2'],
            'categorical_features': ['feature3'],
            'steps': ['imputation', 'scaling'],
            'feature_engineering': {'pca': True}
        },
        'model_type': 'random_forest',
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10
        },
        'evaluation_results': {
            'accuracy': 0.85,
            'precision': 0.83
        }
    }

def test_generate_api_documentation(sample_app, tmp_path):
    """اختبار توليد توثيق API"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(documentation_generator, 'docs_dir', tmp_path)
        
        api_docs = documentation_generator.generate_api_documentation(sample_app)
        
        assert isinstance(api_docs, dict)
        assert 'openapi' in api_docs
        assert 'paths' in api_docs
        assert '/api/test' in api_docs['paths']
        assert '/api/model' in api_docs['paths']
        
        # التحقق من وجود ملف التوثيق
        docs_file = tmp_path / 'api_docs.yaml'
        assert docs_file.exists()
        
        # التحقق من محتوى الملف
        with open(docs_file, 'r', encoding='utf-8') as f:
            saved_docs = yaml.safe_load(f)
            assert saved_docs == api_docs

def test_generate_model_documentation(sample_model_info, tmp_path):
    """اختبار توليد توثيق النموذج"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(documentation_generator, 'docs_dir', tmp_path)
        
        model_docs = documentation_generator.generate_model_documentation(
            'test_model',
            sample_model_info
        )
        
        assert isinstance(model_docs, dict)
        assert 'model_id' in model_docs
        assert 'features' in model_docs
        assert 'preprocessing' in model_docs
        assert 'model' in model_docs
        assert 'performance' in model_docs
        assert 'usage' in model_docs
        
        # التحقق من وجود ملف التوثيق
        docs_file = tmp_path / 'models' / 'test_model_docs.yaml'
        assert docs_file.exists()
        
        # التحقق من محتوى الملف
        with open(docs_file, 'r', encoding='utf-8') as f:
            saved_docs = yaml.safe_load(f)
            assert saved_docs == model_docs

def test_generate_input_format(sample_model_info):
    """اختبار توليد تنسيق المدخلات"""
    input_format = documentation_generator._generate_input_format(sample_model_info)
    
    assert isinstance(input_format, dict)
    assert 'type' in input_format
    assert 'properties' in input_format
    assert all(f in input_format['properties'] 
              for f in sample_model_info['preprocessing_info']['numeric_features'] +
                      sample_model_info['preprocessing_info']['categorical_features'])

def test_generate_output_format(sample_model_info):
    """اختبار ��وليد تنسيق المخرجات"""
    # تصنيف
    output_format = documentation_generator._generate_output_format(sample_model_info)
    assert 'prediction' in output_format['properties']
    assert 'probabilities' in output_format['properties']
    
    # انحدار
    sample_model_info['task_type'] = 'regression'
    output_format = documentation_generator._generate_output_format(sample_model_info)
    assert 'prediction' in output_format['properties']
    assert 'probabilities' not in output_format['properties']

def test_generate_example(sample_model_info):
    """اختبار توليد مثال"""
    example = documentation_generator._generate_example(sample_model_info)
    
    assert isinstance(example, dict)
    assert 'input' in example
    assert 'output' in example
    assert all(f in example['input'] 
              for f in sample_model_info['preprocessing_info']['numeric_features'] +
                      sample_model_info['preprocessing_info']['categorical_features'])

def test_invalid_model_info():
    """اختبار معلومات نموذج غير صالحة"""
    with pytest.raises(Exception):
        documentation_generator.generate_model_documentation(
            'test_model',
            {'invalid': 'info'}
        ) 