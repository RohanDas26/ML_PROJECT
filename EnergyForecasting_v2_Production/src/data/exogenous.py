"""
src/data/exogenous.py

Handles fetching and processing of exogenous variables from FRED and other sources.
Includes caching to avoid redundant network requests.
"""

import io
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

FRED_CONFIG = {
    "industrial_production": "INDPRO",       # Monthly, 1919-present
    "oil_price_wti": "MCOILWTICO",          # Monthly, 1986-present
    "cpi": "CPIAUCSL",                      # Monthly, 1947-present
    "natgas_price": "MHHNGSP",              # Monthly, 1997-present
    "us_population": "B230RC0Q173SBEA",     # Quarterly, 1947-present
    "vehicle_miles_traveled": "TRFVOLUSM227NFWA", # Monthly, 1970-present (Not Seasonally Adjusted)
}

CACHE_DIR = Path("Data/Processed/exogenous_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_fred_series(series_id: str, name: str, start: str = "1973-01-01", end: str = "2025-09-01") -> pd.DataFrame:
    """
    Fetch a single series from FRED. Tries cache first, then network.
    """
    cache_path = CACHE_DIR / f"{name}.csv"
    
    # Try cache
    if cache_path.exists():
        # Check if cache is recent (< 7 days)
        mtime = cache_path.stat().st_mtime
        if time.time() - mtime < 7 * 86400:
            log.info(f"Loading {name} ({series_id}) from cache...")
            df = pd.read_csv(cache_path)
            df["Month"] = pd.to_datetime(df["Month"])
            return df

    # Fetch from network
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}&fq=Monthly&fam=avg"
    )
    log.info(f"Downloading {name} ({series_id}) from FRED...")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["Month", name]
        df["Month"] = pd.to_datetime(df["Month"])
        df[name] = pd.to_numeric(df[name], errors="coerce")
        
        # Save to cache
        df.to_csv(cache_path, index=False)
        return df
        
    except Exception as e:
        log.warning(f"Failed to fetch {name} ({series_id}): {e}")
        return pd.DataFrame(columns=["Month", name])


def compute_hdd_cdd(dates: pd.DatetimeIndex, base_temp: float = 65.0) -> pd.DataFrame:
    """
    Generate realistic monthly HDD/CDD based on a sinusoidal US-average
    temperature model.
    """
    dt_index = pd.DatetimeIndex(dates)
    months = dt_index.month
    
    # Sinusoidal US average temp (F) approx: 34F (Jan) to 74F (Jul)
    # T(t) = 52 + 22 * cos(2*pi*(m-7)/12)
    avg_temp = 52.0 + 22.0 * np.cos(2 * np.pi * (months - 7) / 12.0)
    
    # Add slow climate trend (~+0.03F/year)
    years_since_base = (dt_index.year - 1973) + (dt_index.month - 1) / 12.0
    avg_temp = avg_temp + 0.03 * years_since_base
    
    # Add small random variation
    np.random.seed(42)
    noise = np.random.normal(0, 2.0, len(dt_index))
    avg_temp = avg_temp + noise
    
    days_in_month = dt_index.days_in_month
    hdd = np.maximum(0, base_temp - avg_temp) * days_in_month
    cdd = np.maximum(0, avg_temp - base_temp) * days_in_month
    
    return pd.DataFrame({
        "Month": dates.values if hasattr(dates, 'values') else dates,
        "HDD": np.round(hdd, 0).astype(int),
        "CDD": np.round(cdd, 0).astype(int),
    })


def get_all_exogenous(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Main entry point: Returns DataFrame with all validated exogenous features
    merged on 'Month'. Interpolates missing values.
    """
    # Base DataFrame
    combined = pd.DataFrame({"Month": dates})
    
    # 1. Fetch FRED series
    for name, series_id in FRED_CONFIG.items():
        df_series = fetch_fred_series(series_id, name)
        if not df_series.empty:
            combined = combined.merge(df_series, on="Month", how="left")
            
    # 2. Compute HDD/CDD
    df_weather = compute_hdd_cdd(dates)
    combined = combined.merge(df_weather, on="Month", how="left")
    
    # 3. Interpolation & Backfill
    # Some series like Oil start in 1986. We backfill with the first valid observation 
    # (not ideal but better than dropping). 
    # For Population (quarterly), linear interpolation is perfect.
    cols_to_interp = [c for c in combined.columns if c != "Month"]
    
    for col in cols_to_interp:
        # Linear interp for gaps
        combined[col] = combined[col].interpolate(method="linear")
        # Backfill for pre-start data
        combined[col] = combined[col].bfill()
        # Forwardfill for recent missing
        combined[col] = combined[col].ffill()
        
    return combined
