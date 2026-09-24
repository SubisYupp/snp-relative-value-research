from dataclasses import dataclass, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE_FEATURES = ['dte','is_put','moneyness','iv','atm_iv','relative_iv','local_skew','curvature','surface_residual','forward_variance','iv_change','atm_iv_change','local_skew_change','curvature_change','surface_residual_change','delta','gamma','vega','theta','underlying_return','realized_vol']
REGIME_FEATURES = ['p_low','p_normal','p_high','p_crisis','regime_entropy']
REGIMES = ['LOW','NORMAL','HIGH','CRISIS']

@dataclass
class Config:
    raw_dir: str = 'SNP data'
    seed: int = 42
    horizon_minutes: int = 60
    tolerance_minutes: int = 10
    min_vega: float = 0.01
    warmup_months: int = 2
    neutral_quantile: float = 0.5
    surface_neighbors: int = 13
    n_estimators: int = 600
    early_stopping: int = 35
    version: str = 'snp-rv-v1.1.0'
    processed_dir: str = 'data/processed-v1.1'
    quality_path: str = 'results/preprocessing-v1.1/data_quality.json'

def save_json(path, value):
    import numpy as np
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
        if isinstance(v,(np.integer,)): return int(v)
        if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v,Path): return str(v)
        return v
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,default=str,allow_nan=False),encoding='utf-8')

def calendar_split(timestamps):
    import pandas as pd
    months=sorted(pd.DatetimeIndex(timestamps).to_period('M').unique())
    if len(months)<5: raise ValueError('At least five calendar months required including regime warm-up.')
    n_hold=max(1,round(len(months)*0.09)) if len(months)>16 else 1
    return months[-2*n_hold].start_time, months[-n_hold].start_time
