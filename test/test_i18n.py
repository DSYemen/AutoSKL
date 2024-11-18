import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import yaml
from app.main import create_app
from app.core.config import settings

client = TestClient(create_app())

@pytest.fixture
def translations():
    """تحميل ملفات الترجمة"""
    translations_dir = Path('locales')
    translations = {}
    
    for lang_file in translations_dir.glob('*.json'):
        lang = lang_file.stem
        with open(lang_file, 'r', encoding='utf-8') as f:
            translations[lang] = json.load(f)
            
    return translations

def test_supported_languages(translations):
    """اختبار اللغات المدعومة"""
    # التحقق من دعم اللغات الأساسية
    assert 'ar' in translations
    assert 'en' in translations
    
    # التحقق من اكتمال الترجمات
    ar_keys = set(translations['ar'].keys())
    en_keys = set(translations['en'].keys())
    assert ar_keys == en_keys

def test_arabic_content():
    """اختبار المحتوى العربي"""
    response = client.get("/", headers={"Accept-Language": "ar"})
    assert response.status_code == 200
    
    # التحقق من وجود نص عربي
    assert 'dir="rtl"' in response.text
    assert 'lang="ar"' in response.text
    assert 'إطار عمل التعلم الآلي' in response.text

def test_english_content():
    """اختبار المحتوى الإنجليزي"""
    response = client.get("/", headers={"Accept-Language": "en"})
    assert response.status_code == 200
    
    # التحقق من وجود نص إنجليزي
    assert 'dir="ltr"' in response.text
    assert 'lang="en"' in response.text
    assert 'Machine Learning Framework' in response.text

def test_error_messages(translations):
    """اختبار رسائل الخطأ"""
    for lang, trans in translations.items():
        response = client.get(
            "/api/models/nonexistent",
            headers={"Accept-Language": lang}
        )
        
        error_msg = response.json()["detail"]
        assert error_msg == trans["model_not_found"]

def test_number_formatting():
    """اختبار تنسيق الأرقام"""
    from babel.numbers import format_number
    
    number = 1234.5678
    
    # تنسيق عربي
    ar_format = format_number(number, locale='ar')
    assert ',' not in ar_format  # الفاصلة العشرية العربية
    assert '٫' in ar_format
    
    # تنسيق إنجليزي
    en_format = format_number(number, locale='en')
    assert '.' in en_format
    assert ',' in en_format

def test_date_formatting():
    """اختبار تنسيق التواريخ"""
    from babel.dates import format_date
    from datetime import date
    
    test_date = date(2024, 1, 1)
    
    # تنسيق عربي
    ar_date = format_date(test_date, locale='ar')
    assert 'يناير' in ar_date
    
    # تنسيق إنجليزي
    en_date = format_date(test_date, locale='en')
    assert 'January' in en_date

def test_rtl_support():
    """اختبار دعم الكتابة من اليمين لليسار"""
    response = client.get("/", headers={"Accept-Language": "ar"})
    
    # التحقق من وجود الأنماط الخاصة بـ RTL
    assert 'text-align: right' in response.text
    assert 'float: right' in response.text
    assert 'margin-left' in response.text

def test_translation_fallback():
    """اختبار الرجوع للغة الافتراضية"""
    # طلب لغة غير مدعومة
    response = client.get("/", headers={"Accept-Language": "fr"})
    
    # يجب أن يرجع للغة الافتراضية
    assert response.status_code == 200
    assert settings.default_language in response.text

def test_translation_interpolation(translations):
    """اختبار تضمين المتغيرات في النصوص"""
    for lang, trans in translations.items():
        if 'welcome_user' in trans:
            message = trans['welcome_user'].format(username='test')
            assert 'test' in message

def test_pluralization():
    """اختبار صيغ الجمع"""
    from babel.plural import PluralRule
    
    # قواعد الجمع العربية
    ar_plural = PluralRule.parse('zero: n=0; one: n=1; two: n=2; few: n%100=3..10; many: n%100=11..99; other')
    
    test_numbers = [0, 1, 2, 3, 11, 100]
    categories = [ar_plural(n) for n in test_numbers]
    assert len(set(categories)) > 1  # التأكد من وجود صيغ مختلفة

def test_bidirectional_text():
    """اختبار النصوص ثنائية الاتجاه"""
    response = client.get("/", headers={"Accept-Language": "ar"})
    
    # التحقق من وجود علامات التحكم في اتجاه النص
    assert '&#x202B;' in response.text  # RLE
    assert '&#x202A;' in response.text  # LRE
    assert '&#x202C;' in response.text  # PDF

def test_currency_formatting():
    """اختبار تنسيق العملات"""
    from babel.numbers import format_currency
    
    amount = 1234.56
    
    # تنسيق بالريال السعودي
    sar_ar = format_currency(amount, 'SAR', locale='ar')
    assert 'ر.س' in sar_ar
    
    # تنسيق بالدولار
    usd_en = format_currency(amount, 'USD', locale='en')
    assert '$' in usd_en

def test_translation_files():
    """اختبار ملفات الترجمة"""
    locales_dir = Path('locales')
    
    # التحقق من وجود ملفات الترجمة
    assert (locales_dir / 'ar.json').exists()
    assert (locales_dir / 'en.json').exists()
    
    # التحقق من صحة تنسيق JSON
    for lang_file in locales_dir.glob('*.json'):
        with open(lang_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            assert isinstance(content, dict) 