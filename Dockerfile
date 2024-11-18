# المرحلة الأولى: بناء التطبيق
FROM python:3.12-slim as builder

WORKDIR /app

# نسخ ملفات التبعيات
COPY requirements.txt pyproject.toml ./

# تثبيت التبعيات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود التطبيق
COPY . .

# المرحلة الثانية: تشغيل التطبيق
FROM python:3.12-slim

WORKDIR /app

# نسخ التبعيات والكود من المرحلة السابقة
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app .

# إنشاء مستخدم غير root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# تعريف المتغيرات البيئية
ENV PYTHONPATH=/app
ENV PORT=8000

# كشف المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 