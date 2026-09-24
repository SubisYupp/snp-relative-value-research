"""Integration checks on the frozen experiment, without model refitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytestmark=pytest.mark.skipif(not Path('results/artifacts/test_evaluation_complete.json').exists(),reason='Actual-data pipeline has not completed')

def test_frozen_model_identity():
    lock=json.loads(Path('results/artifacts/design_lock.json').read_text())
    assert lock['model_sha256']==hashlib.sha256(Path('models/xgboost/with_regime.json').read_bytes()).hexdigest()
    assert pd.Timestamp(lock['train_end'])<pd.Timestamp(lock['validation_start'])
    assert pd.Timestamp(lock['validation_end'])<pd.Timestamp(lock['test_start'])

def test_historical_rerun_preserves_previous_model_and_labels_output():
    lock=json.loads(Path('results/artifacts/design_lock.json').read_text())
    if not lock.get('historical_reevaluation'): pytest.skip('Original experiment, not a rerun')
    archive=Path('experiments/snp-rv-v1.0.0')
    previous=json.loads((archive/'results/artifacts/design_lock.json').read_text())
    assert hashlib.sha256((archive/'models/xgboost/with_regime.json').read_bytes()).hexdigest()==previous['model_sha256']
    summary=json.loads(Path('dashboard/public/data/summary.json').read_text())
    assert summary['lock']['historical_reevaluation']
    assert 'not a fresh untouched' in summary['conclusions'][0]

def test_saved_predictions_and_shap_reconstruct():
    d=pd.read_parquet('results/predictions/predictions_test.parquet')
    shap=np.load('results/artifacts/shap_test.npy'); meta=json.loads(Path('results/artifacts/explanation.json').read_text())
    np.testing.assert_allclose(meta['baseline']+shap.sum(axis=1),d.prediction,atol=1e-4,rtol=1e-4)
    assert d.loc[d.target.notna(),'actual_horizon_minutes'].between(50,70).all()
    assert (d.loc[d.target.notna(),'target_timestamp']>d.loc[d.target.notna(),'timestamp']).all()
    assert np.isfinite(d.prediction).all()

def test_dashboard_metrics_equal_saved_results():
    exported=json.loads(Path('dashboard/public/data/summary.json').read_text()); actual=json.loads(Path('results/metrics/metrics.json').read_text())
    assert exported['metrics']==actual
    assert sum(r['rows'] for r in exported['snapshots'])==exported['test_total_rows']
    q=exported['quality']; assert q['cleaned_rows']+sum(q['drops'].values())==q['input_option_rows']

def test_no_cross_boundary_labels_and_signed_put_deltas():
    d=pd.read_parquet('results/predictions/predictions_validation.parquet')
    assert (d.target_timestamp<pd.Timestamp('2026-09-01')).all()
    assert (d[d.option_type=='P'].delta<=0).all()
    assert not d.duplicated(['contract_id','timestamp']).any()
