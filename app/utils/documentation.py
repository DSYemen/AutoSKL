from typing import Dict, Any, List, Optional, Protocol, TypeVar
import yaml
from pathlib import Path
import inspect
from app.core.config import settings
import logging
from fastapi.openapi.utils import get_openapi
import json
from datetime import datetime
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# تعريف الأنواع المخصصة
DocData: TypeAlias = Dict[str, Any]
ModelDocs: TypeAlias = Dict[str, Any]

class DocumentationGenerator:
    """مولد التوثيق"""
    def __init__(self) -> None:
        self.docs_dir = Path("docs")
        self.docs_dir.mkdir(exist_ok=True)
        self.api_version = settings.app.version
        self.last_update = datetime.now()
        
    def generate_api_documentation(self, app) -> DocData:
        """توليد توثيق API محسن"""
        try:
            # توليد مواصفات OpenAPI
            openapi_schema = get_openapi(
                title=settings.app.name,
                version=self.api_version,
                description="توثيق API لإطار عمل التعلم الآلي",
                routes=app.routes
            )
            
            # إضافة معلومات إضافية
            openapi_schema.update({
                'info': {
                    'title': settings.app.name,
                    'version': self.api_version,
                    'description': 'توثيق API لإطار عمل التعلم الآلي',
                    'contact': {
                        'name': 'فريق التطوير',
                        'email': settings.email.sender
                    },
                    'license': {
                        'name': 'MIT',
                        'url': 'https://opensource.org/licenses/MIT'
                    }
                },
                'servers': [
                    {
                        'url': f'http://{settings.app.host}:{settings.app.port}',
                        'description': 'خادم التطوير'
                    }
                ]
            })
            
            # حفظ التوثيق
            docs_path = self.docs_dir / 'openapi.json'
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(openapi_schema, f, ensure_ascii=False, indent=2)
                
            logger.info(f"تم توليد توثيق API في {docs_path}")
            return openapi_schema
            
        except Exception as e:
            logger.error(f"خطأ في توليد توثيق API: {str(e)}")
            raise
            
    def generate_model_documentation(self, model_info: DocData) -> ModelDocs:
        """توليد توثيق النموذج"""
        try:
            model_docs = {
                'model_id': model_info['model_id'],
                'task_type': model_info['task_type'],
                'description': model_info.get('description', ''),
                'features': self._generate_feature_documentation(model_info),
                'preprocessing': self._generate_preprocessing_documentation(model_info),
                'performance': self._generate_performance_documentation(model_info),
                'usage': self._generate_usage_documentation(model_info),
                'metadata': {
                    'created_at': model_info.get('creation_date'),
                    'last_updated': model_info.get('last_updated'),
                    'version': model_info.get('version')
                }
            }
            
            # حفظ التوثيق
            model_docs_path = self.docs_dir / f"model_{model_info['model_id']}.yaml"
            with open(model_docs_path, 'w', encoding='utf-8') as f:
                yaml.dump(model_docs, f, default_flow_style=False, allow_unicode=True)
                
            logger.info(f"تم توليد توثيق النموذج في {model_docs_path}")
            return model_docs
            
        except Exception as e:
            logger.error(f"خطأ في توليد توثيق النموذج: {str(e)}")
            raise
            
    def _generate_feature_documentation(self, model_info: DocData) -> Dict[str, Any]:
        """توليد توثيق الميزات"""
        features_info = model_info.get('preprocessing_info', {})
        return {
            'numeric_features': features_info.get('numeric_features', []),
            'categorical_features': features_info.get('categorical_features', []),
            'target_column': model_info.get('target_column'),
            'feature_importance': model_info.get('feature_importance', {})
        }
        
    def _generate_preprocessing_documentation(self, model_info: DocData) -> Dict[str, Any]:
        """توليد توثيق معالجة البيانات"""
        preprocessing_info = model_info.get('preprocessing_info', {})
        return {
            'scaling_method': preprocessing_info.get('scaling', {}).get('method'),
            'encoding_method': preprocessing_info.get('encoding', {}).get('method'),
            'missing_value_strategy': preprocessing_info.get('missing_value_strategy'),
            'feature_selection': preprocessing_info.get('feature_selection', {})
        }
        
    def _generate_performance_documentation(self, model_info: DocData) -> Dict[str, Any]:
        """توليد توثيق الأداء"""
        evaluation_results = model_info.get('evaluation_results', {})
        return {
            'metrics': evaluation_results.get('metrics', {}),
            'validation_strategy': evaluation_results.get('validation_strategy'),
            'training_time': evaluation_results.get('training_time'),
            'last_evaluation': evaluation_results.get('last_evaluation_time')
        }
        
    def _generate_usage_documentation(self, model_info: DocData) -> Dict[str, Any]:
        """توليد توثيق الاستخدام"""
        return {
            'input_format': self._generate_input_format(model_info),
            'output_format': self._generate_output_format(model_info),
            'example': self._generate_example(model_info)
        }
        
    def _generate_input_format(self, model_info: DocData) -> Dict[str, Any]:
        """توليد تنسيق المدخلات"""
        preprocessing_info = model_info.get('preprocessing_info', {})
        return {
            'type': 'object',
            'properties': {
                feature: {
                    'type': 'number' if feature in preprocessing_info.get('numeric_features', []) else 'string',
                    'description': f"Feature: {feature}"
                }
                for feature in preprocessing_info.get('numeric_features', []) + 
                              preprocessing_info.get('categorical_features', [])
            }
        }
        
    def _generate_output_format(self, model_info: DocData) -> Dict[str, Any]:
        """توليد تنسيق المخرجات"""
        if model_info['task_type'] == 'classification':
            return {
                'type': 'object',
                'properties': {
                    'prediction': {
                        'type': 'string',
                        'description': 'الفئة المتوقعة'
                    },
                    'probabilities': {
                        'type': 'object',
                        'description': 'احتمالات كل فئة'
                    }
                }
            }
        else:
            return {
                'type': 'object',
                'properties': {
                    'prediction': {
                        'type': 'number',
                        'description': 'القيمة المتوقعة'
                    }
                }
            }
            
    def _generate_example(self, model_info: DocData) -> Dict[str, Any]:
        """توليد مثال للاستخدام"""
        preprocessing_info = model_info.get('preprocessing_info', {})
        return {
            'input': {
                feature: 0.0 if feature in preprocessing_info.get('numeric_features', []) else 'example'
                for feature in preprocessing_info.get('numeric_features', []) + 
                              preprocessing_info.get('categorical_features', [])
            },
            'output': {
                'prediction': 'example_class' if model_info['task_type'] == 'classification' else 0.0,
                'probabilities': {'class_1': 0.8, 'class_2': 0.2} if model_info['task_type'] == 'classification' else None
            }
        }

documentation_generator = DocumentationGenerator() 