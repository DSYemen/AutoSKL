from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, field_validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
import os
from functools import lru_cache

class AppSettings(BaseSettings):
    """إعدادات التطبيق الأساسية"""
    name: str = "AutoML Framework"
    version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = Field(default="your-secret-key-here")
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class DatabaseSettings(BaseSettings):
    """إعدادات قاعدة البيانات"""
    url: str = Field(default="sqlite:///./ml_framework.db")
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    echo: bool = False
    statement_timeout: int = 30000  # بالمللي ثانية
    idle_timeout: int = 60000  # بالمللي ثانية
    connection_retries: int = 3
    retry_interval: int = 5
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

    @field_validator('url')
    def validate_database_url(cls, v: str) -> str:
        """التحقق من صحة عنوان قاعدة البيانات"""
        if not v:
            raise ValueError("عنوان قاعدة البيانات مطلوب")
        return v

class RedisSettings(BaseSettings):
    """إعدادات Redis"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    encoding: str = "utf-8"
    decode_responses: bool = True
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

    @field_validator('port')
    def validate_port(cls, v: int) -> int:
        """التحقق من صحة رقم المنفذ"""
        if not 1 <= v <= 65535:
            raise ValueError("رقم المنفذ يجب أن يكون بين 1 و 65535")
        return v

class EmailSettings(BaseSettings):
    """إعدادات البريد الإلكتروني"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender: str = "your-email@example.com"
    password: str = ""
    use_tls: bool = True
    timeout: int = 60
    recipients: List[str] = []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class LoggingSettings(BaseSettings):
    """إعدادات التسجيل"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    json_format: bool = True
    log_sql: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class ModelSelectionSettings(BaseSettings):
    """إعدادات اختيار النموذج"""
    cv_folds: int = 5
    n_trials: int = 200
    timeout: int = 3600
    metric: Dict[str, str] = {
        "classification": "accuracy",
        "regression": "neg_mean_squared_error",
        "clustering": "silhouette"
    }
    early_stopping: Dict[str, Any] = {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001
    }
    pruning: Dict[str, Any] = {
        "enabled": True,
        "n_startup_trials": 5,
        "n_warmup_steps": 10
    }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class TrainingSettings(BaseSettings):
    """إعدادات التدريب"""
    test_size: float = 0.2
    random_state: int = 42
    min_samples: int = 100
    improvement_threshold: float = 0.01
    max_training_time: int = 7200
    validation_split: float = 0.1
    shuffle: bool = True
    stratify: bool = True
    batch_size: int = 32
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class MonitoringSettings(BaseSettings):
    """إعدادات المراقبة"""
    drift_threshold: float = 0.05
    performance_threshold: float = 0.95
    check_interval: int = 3600
    metrics_history_days: int = 30
    alert_cooldown: int = 3600
    min_samples_drift: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class MLSettings(BaseSettings):
    """إعدادات التعلم الآلي"""
    model_config = SettingsConfigDict(protected_namespaces=('settings_',))
    
    task_types: List[str] = [
        "classification",
        "regression",
        "clustering",
        "dimensionality_reduction"
    ]
    
    model_selection: ModelSelectionSettings = Field(default_factory=ModelSelectionSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

class CacheSettings(BaseSettings):
    """إعدادات التخزين المؤقت"""
    type: str = "redis"
    default_ttl: int = 3600
    key_prefix: str = "mlf:"
    serializer: str = "json"
    compression: bool = True
    max_memory: str = "1gb"
    eviction_policy: str = "allkeys-lru"
    local_cache_size: int = 1000
    local_cache_ttl: int = 300
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class StorageSettings(BaseSettings):
    """إعدادات التخزين"""
    type: str = "local"
    models_dir: str = "models"
    reports_dir: str = "reports"
    temp_dir: str = "temp"
    backup_dir: str = "backups"
    max_file_size: int = 104857600  # 100MB

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class AlertSettings(BaseSettings):
    """إعدادات التنبيهات"""
    enabled: bool = True
    recipients: List[str] = []
    cooldown_period: int = 3600
    severity_levels: Dict[str, int] = {
        "critical": 1,
        "high": 2,
        "medium": 3,
        "low": 4
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class CelerySettings(BaseSettings):
    """إعدادات Celery"""
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = ["json"]
    timezone: str = "UTC"
    enable_utc: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class ReportingSettings(BaseSettings):
    """إعدادات التقارير"""
    template_dir: str = "templates/reports"
    output_dir: str = "reports"
    default_format: str = "pdf"
    max_plots: int = 10
    dpi: int = 300
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow'
    )

class Settings(BaseSettings):
    """الإعدادات الرئيسية"""
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='allow',
        protected_namespaces=()
    )
    
    @classmethod
    def load_from_yaml(cls, yaml_file: str = "config.yaml") -> "Settings":
        """تحميل الإعدادات من ملف YAML"""
        try:
            if Path(yaml_file).exists():
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_settings = yaml.safe_load(f)
                    # دمج الإعدادات من الملف مع الإعدادات البيئية
                    env_settings = cls()
                    if yaml_settings:
                        yaml_settings.update(env_settings.model_dump())
                        return cls(**yaml_settings)
            return cls()
        except Exception as e:
            print(f"خطأ في تحميل ملف الإعدادات: {str(e)}")
            return cls()

@lru_cache()
def get_settings() -> Settings:
    """الحصول على نسخة مخزنة مؤقتاً من الإعدادات"""
    return Settings.load_from_yaml()

# تحميل الإعدادات
settings = get_settings()

# التحقق من وجود المجلدات المطلوبة
required_dirs = ['logs', 'models', 'data', 'reports', 'temp', 'backups']
for dir_name in required_dirs:
    Path(dir_name).mkdir(exist_ok=True) 