"""Checks for the separate preprocessing release; never rescore the old model."""
import json
from pathlib import Path
import pandas as pd
import pytest
from quant.config import Config
from quant.preprocess import pipeline_fingerprint

pytestmark=pytest.mark.skipif(not Path(Config().quality_path).exists(),reason='Revised preprocessing has not run')

def test_revised_output_is_separate_and_accounts_for_every_row():
    config=Config();quality=json.loads(Path(config.quality_path).read_text())
    assert Path(config.processed_dir).resolve()!=Path('data/processed').resolve()
    assert quality['cleaned_rows']+sum(quality['drops'].values())==quality['input_option_rows']
    assert quality['iv_above_300pct_retained_flag']==0
    assert quality['pipeline_fingerprint']==pipeline_fingerprint(config)
    assert len(quality['source_manifest'])==len(quality['processed_files'])==324

def test_corrupt_training_price_and_zero_quotes_not_used():
    path=Path(Config().processed_dir)
    d=pd.read_parquet(path/'2025-10-10.parquet')
    suspect=(d.contract=='ODSNP5DV25')&(d.strike==6420)&(d.option_type=='C')
    assert not (d.loc[suspect,'timestamp']==pd.Timestamp('2025-10-10 10:00')).any()
    prior=d.loc[suspect&(d.timestamp==pd.Timestamp('2025-10-10 09:00'))]
    assert len(prior)==1 and prior.target.isna().all()
    zeros=pd.read_parquet(path/'2026-03-26.parquet')
    assert (zeros.option_price>0).all()

def test_review_asserts_old_model_preserved_and_cache_valid():
    report=json.loads(Path('reports/preprocessing_review.json').read_text())
    assert report['original_model_unchanged'] and not report['retrained']
    for k in ['revised_infinite_numerics','revised_negative_expected_iv','revised_duplicate_identities']:
        assert report['comparison'][k]==0
