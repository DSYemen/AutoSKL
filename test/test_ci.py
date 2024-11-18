import pytest
import subprocess
import os
import sys
from pathlib import Path
import yaml
import json
import re

def test_project_structure():
    """اختبار هيكل المشروع"""
    required_dirs = [
        'app',
        'tests',
        'docs',
        'locales',
        'static',
        'templates'
    ]
    
    for dir_name in required_dirs:
        assert Path(dir_name).is_dir(), f"المجلد {dir_name} غير موجود"

def test_code_style():
    """اختبار نمط الكود"""
    # التحقق من التنسيق باستخدام black
    result = subprocess.run(['black', '--check', '.'], capture_output=True)
    assert result.returncode == 0, "يجب تنسيق الكود باستخدام black"
    
    # التحقق من الأخطاء باستخدام flake8
    result = subprocess.run(['flake8', '.'], capture_output=True)
    assert result.returncode == 0, "يوجد أخطاء في نمط الكود"

def test_type_hints():
    """اختبار تلميحات النوع"""
    # التحقق من التلميحات باستخدام mypy
    result = subprocess.run(['mypy', '.'], capture_output=True)
    assert result.returncode == 0, "يوجد أخطاء في تلميحات النوع"

def test_test_coverage():
    """اختبار تغطية الاختبارات"""
    # تشغيل الاختبارات مع تقرير التغطية
    result = subprocess.run(
        ['pytest', '--cov=app', '--cov-report=term-missing'],
        capture_output=True
    )
    
    # استخراج نسبة التغطية
    coverage_output = result.stdout.decode()
    coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', coverage_output)
    coverage = int(coverage_match.group(1))
    
    assert coverage >= 80, "تغطية الاختبارات يجب أن تكون 80% على الأقل"

def test_dependencies():
    """اختبار التبعيات"""
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    
    # التحقق من تثبيت جميع التبعيات
    for req in requirements:
        if req and not req.startswith('#'):
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', req.split('==')[0]],
                capture_output=True
            )
            assert result.returncode == 0, f"التبعية {req} غير مثبتة"

def test_documentation():
    """اختبار التوثيق"""
    docs_dir = Path('docs')
    
    # التحقق من وجود ملفات التوثيق الأساسية
    assert (docs_dir / 'README.md').exists()
    assert (docs_dir / 'API.md').exists()
    
    # التحقق من صحة Markdown
    result = subprocess.run(['mdl', 'docs'], capture_output=True)
    assert result.returncode == 0, "يوجد أخطاء في تنسيق Markdown"

def test_git_hooks():
    """اختبار Git hooks"""
    hooks_dir = Path('.git/hooks')
    
    # التحقق من وجود hooks مطلوبة
    required_hooks = ['pre-commit', 'pre-push']
    for hook in required_hooks:
        assert (hooks_dir / hook).exists(), f"Git hook {hook} غير موجود"
        assert os.access(hooks_dir / hook, os.X_OK), f"Git hook {hook} غير قابل للتنفيذ"

def test_docker_setup():
    """اختبار إعداد Docker"""
    # التحقق من وجود ملفات Docker
    assert Path('Dockerfile').exists()
    assert Path('docker-compose.yml').exists()
    
    # التحقق من صحة Dockerfile
    result = subprocess.run(['hadolint', 'Dockerfile'], capture_output=True)
    assert result.returncode == 0, "يوجد أخطاء في Dockerfile"

def test_ci_config():
    """اختبار تكوين CI"""
    # التحقق من تكوين GitHub Actions
    github_dir = Path('.github/workflows')
    assert github_dir.exists()
    
    for workflow in github_dir.glob('*.yml'):
        with open(workflow, 'r') as f:
            config = yaml.safe_load(f)
            assert 'jobs' in config, f"تكوين غير صالح في {workflow}"

def test_security_checks():
    """اختبار فحوصات الأمان"""
    # فحص الثغرات الأمنية في التبعيات
    result = subprocess.run(['safety', 'check'], capture_output=True)
    assert result.returncode == 0, "يوجد ثغرات أمنية في التبعيات"
    
    # فحص الكود باستخدام bandit
    result = subprocess.run(['bandit', '-r', 'app'], capture_output=True)
    assert result.returncode == 0, "يوجد مشاكل أمنية في الكود"

def test_environment_variables():
    """اختبار متغيرات البيئة"""
    # التحقق من وجود ملف .env.example
    assert Path('.env.example').exists()
    
    # التحقق من عدم تتبع ملف .env
    gitignore = Path('.gitignore').read_text()
    assert '.env' in gitignore, "يجب عدم تتبع ملف .env"

def test_license():
    """اختبار الترخيص"""
    # التحقق من وجود ملف الترخيص
    assert Path('LICENSE').exists()
    
    # التحقق من نوع الترخيص
    license_text = Path('LICENSE').read_text()
    assert any(license_type in license_text 
              for license_type in ['MIT', 'Apache', 'GPL'])

def test_changelog():
    """اختبار سجل التغييرات"""
    # التحقق من وجود وتنسيق CHANGELOG
    changelog = Path('CHANGELOG.md')
    assert changelog.exists()
    
    content = changelog.read_text()
    assert re.search(r'## \[\d+\.\d+\.\d+\]', content), "تنسيق غير صالح للإصدارات"

def test_build_artifacts():
    """اختبار مخرجات البناء"""
    # التحقق من تجاهل مجلدات البناء
    gitignore = Path('.gitignore').read_text()
    build_patterns = ['__pycache__', '*.pyc', 'build/', 'dist/']
    for pattern in build_patterns:
        assert pattern in gitignore, f"يجب تجاهل {pattern}" 