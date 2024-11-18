import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
from app.db.database import engine, Base, get_db, create_db_and_tables
from app.db.models import (
    Model, Prediction, ModelMetrics, ModelUpdate,
    DataDrift, TrainingJob, FeatureImportance
)

@pytest.fixture
async def db_session():
    """جلسة قاعدة بيانات للاختبار"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()

@pytest.fixture
async def sample_model(db_session):
    """نموذج للاختبار"""
    model = Model(
        model_id='test_model',
        task_type='classification',
        target_column='target',
        metadata={
            'feature_names': ['f1', 'f2'],
            'preprocessing_info': {'steps': ['scaling']}
        }
    )
    db_session.add(model)
    await db_session.commit()
    return model

async def test_create_tables():
    """اختبار إنشاء الجداول"""
    await create_db_and_tables()
    
    # التحقق من وجود الجداول
    async with engine.begin() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: sync_conn.dialect.get_table_names(sync_conn)
        )
        
        assert 'models' in tables
        assert 'predictions' in tables
        assert 'model_metrics' in tables
        assert 'model_updates' in tables
        assert 'data_drift' in tables
        assert 'training_jobs' in tables
        assert 'feature_importance' in tables

async def test_model_crud(db_session):
    """اختبار عمليات CRUD للنموذج"""
    # إنشاء
    model = Model(
        model_id='test_model',
        task_type='classification',
        target_column='target'
    )
    db_session.add(model)
    await db_session.commit()
    
    # قراءة
    stmt = select(Model).where(Model.model_id == 'test_model')
    result = await db_session.execute(stmt)
    saved_model = result.scalar_one()
    assert saved_model.model_id == 'test_model'
    
    # تحديث
    saved_model.metadata = {'updated': True}
    await db_session.commit()
    
    # حذف
    await db_session.delete(saved_model)
    await db_session.commit()
    
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None

async def test_predictions(db_session, sample_model):
    """اختبار سجلات التنبؤات"""
    prediction = Prediction(
        model_id=sample_model.model_id,
        input_data={'f1': 1, 'f2': 2},
        prediction={'class': 1},
        confidence=0.85
    )
    db_session.add(prediction)
    await db_session.commit()
    
    stmt = select(Prediction).where(Prediction.model_id == sample_model.model_id)
    result = await db_session.execute(stmt)
    saved_prediction = result.scalar_one()
    
    assert saved_prediction.confidence == 0.85
    assert saved_prediction.prediction['class'] == 1

async def test_model_metrics(db_session, sample_model):
    """اختبار مقاييس النموذج"""
    metrics = ModelMetrics(
        model_id=sample_model.model_id,
        metrics={'accuracy': 0.85}
    )
    db_session.add(metrics)
    await db_session.commit()
    
    stmt = select(ModelMetrics).where(ModelMetrics.model_id == sample_model.model_id)
    result = await db_session.execute(stmt)
    saved_metrics = result.scalar_one()
    
    assert saved_metrics.metrics['accuracy'] == 0.85

async def test_model_update(db_session, sample_model):
    """اختبار تحديثات النموذج"""
    update = ModelUpdate(
        model_id=sample_model.model_id,
        performance_before={'accuracy': 0.8},
        performance_after={'accuracy': 0.85}
    )
    db_session.add(update)
    await db_session.commit()
    
    stmt = select(ModelUpdate).where(ModelUpdate.model_id == sample_model.model_id)
    result = await db_session.execute(stmt)
    saved_update = result.scalar_one()
    
    assert saved_update.performance_before['accuracy'] == 0.8
    assert saved_update.performance_after['accuracy'] == 0.85

async def test_data_drift(db_session, sample_model):
    """اختبار انحراف البيانات"""
    drift = DataDrift(
        model_id=sample_model.model_id,
        feature_name='f1',
        drift_score=0.1,
        drift_detected=True
    )
    db_session.add(drift)
    await db_session.commit()
    
    stmt = select(DataDrift).where(DataDrift.model_id == sample_model.model_id)
    result = await db_session.execute(stmt)
    saved_drift = result.scalar_one()
    
    assert saved_drift.feature_name == 'f1'
    assert saved_drift.drift_detected is True

async def test_training_job(db_session, sample_model):
    """اختبار مهام التدريب"""
    job = TrainingJob(
        model_id=sample_model.model_id,
        status='running'
    )
    db_session.add(job)
    await db_session.commit()
    
    stmt = select(TrainingJob).where(TrainingJob.model_id == sample_model.model_id)
    result = await db_session.execute(stmt)
    saved_job = result.scalar_one()
    
    assert saved_job.status == 'running'
    assert saved_job.end_time is None

async def test_feature_importance(db_session, sample_model):
    """اختبار أهمية الميزات"""
    importance = FeatureImportance(
        model_id=sample_model.model_id,
        feature_name='f1',
        importance_score=0.7
    )
    db_session.add(importance)
    await db_session.commit()
    
    stmt = select(FeatureImportance).where(
        FeatureImportance.model_id == sample_model.model_id
    )
    result = await db_session.execute(stmt)
    saved_importance = result.scalar_one()
    
    assert saved_importance.feature_name == 'f1'
    assert saved_importance.importance_score == 0.7 