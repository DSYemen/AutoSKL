# Python Starter

Quickly get started with [Python](https://www.python.org/) using this starter!

- If you want to upgrade Python, you can change the image in the [Dockerfile](./.devcontainer/Dockerfile).

## Requirements.txt

```
fastapi==0.110.0
uvicorn==0.27.1
python-multipart==0.0.9
pydantic==2.6.1
pydantic-settings==2.1.0
python-jose==3.3.0
python-dotenv==1.0.1
httpx==0.26.0
sse-starlette==1.8.2
sqlalchemy==2.0.25
aiosqlite==0.19.0
asyncpg==0.29.0
alembic==1.13.1
redis==5.0.1
celery==5.3.6
flower==2.0.1
apscheduler==3.10.4
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.2
optuna==3.5.0
shap==0.44.0
plotly==5.18.0
kaleido==0.2.1
seaborn==0.13.2
matplotlib==3.8.3
jinja2==3.1.2
pdfkit==1.0.0
prometheus-client==0.20.0
psutil==5.9.8
aiofiles==23.2.1
python-json-logger==2.0.7
pytest==8.0.0
pytest-asyncio==0.23.5
pytest-cov==4.1.0
black==24.2.0
isort==5.13.2
mypy==1.8.0
flake8==7.0.0
sphinx==7.2.6
passlib==1.7.4
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
python-multipart==0.0.9
tqdm==4.66.2
joblib==1.3.2
PyYAML==6.0.1
python-dateutil==2.8.2
pytz==2024.1
aiosmtplib==3.0.1
uvloop==0.19.0
orjson==3.9.15
email-validator==2.1.0.post1
python-magic==0.4.27
```

### to install

python 3.12

### update pip

```
pip install --upgrade pip
```


and Requirements.txt

```
 pip install --upgrade -r requirements.txt
```

file Requirements.txt

```
fastapi
uvicorn
python-multipart
pydantic
pydantic-settings
python-jose
python-dotenv
httpx
sse-starlette
sqlalchemy
aiosqlite
asyncpg
alembic
redis
celery
flower
apscheduler
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
plotly
kaleido
seaborn
matplotlib
jinja2
pdfkit
prometheus-client
psutil
aiofiles
python-json-logger
pytest
pytest-asyncio
pytest-cov
black
isort
mypy
flake8
sphinx
passlib
bcrypt
python-jose[cryptography]
python-multipart
tqdm
joblib
PyYAML
python-dateutil
pytz
aiosmtplib
uvloop
orjson
email-validator
python-magic
aioredis
```

```
# إطار العمل الأساسي
fastapi==0.110.0
uvicorn[standard]==0.27.1
python-multipart==0.0.9
pydantic==2.6.1
pydantic-settings==2.1.0
python-jose[cryptography]==3.3.0
python-dotenv==1.0.1
httpx==0.26.0
sse-starlette==1.8.2

# قاعدة البيانات
sqlalchemy[asyncio]==2.0.25
aiosqlite==0.19.0
asyncpg==0.29.0
alembic==1.13.1

# التخزين المؤقت والمهام
redis==5.0.1
celery==5.3.6
flower==2.0.1
apscheduler==3.10.4

# معالجة البيانات والتعلم الآلي
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.2
optuna==3.5.0
shap==0.44.0

# التصور والتقارير
plotly==5.18.0
kaleido==0.2.1
seaborn==0.13.2
matplotlib==3.8.3
jinja2==3.1.2
pdfkit==1.0.0

# المراقبة والقياس
prometheus-client==0.20.0
psutil==5.9.8

# الملفات والتخزين
aiofiles==23.2.1
python-json-logger==2.0.7

# الاختبار والتطوير
pytest==8.0.0
pytest-asyncio==0.23.5
pytest-cov==4.1.0
black==24.2.0
isort==5.13.2
mypy==1.8.0
flake8==7.0.0
sphinx==7.2.6

# الأمان والمصادقة
passlib==1.7.4
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
python-multipart==0.0.9

# أدوات إضافية
tqdm==4.66.2
joblib==1.3.2
PyYAML==6.0.1
python-dateutil==2.8.2
pytz==2024.1
aiosmtplib==3.0.1

# تحسينات الأداء
uvloop==0.19.0 ; sys_platform != 'win32'  # لا يدعم Windows
orjson==3.9.15
email-validator==2.1.0.post1
python-magic==0.4.27

# تبعيات اختيارية لتحسين الأداء
# hiredis==2.3.2  # لتحسين أداء Redis
# ujson==5.9.0    # لتحسين أداء معالجة JSON
# cchardet==2.1.7 # لتحسين أداء معالجة النصوص
# aiodns==3.1.1   # لتحسين أداء DNS
# Brotli==1.1.0   # لضغط HTTP

```
