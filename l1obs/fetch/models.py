"""Provider-independent retrieval result."""
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class FetchResult:
    df: pd.DataFrame
    dataset_id: str
    used_vars: Dict[str, str]
