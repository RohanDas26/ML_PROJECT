"""
system_run.py — Intensive Model Verification Suite
=====================================================
Runs the full pipeline (load → features → split → scale → train → forecast)
across multiple sectors and horizons to verify model stability.

Usage:
    python system_run.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data, validate_data, ALL_SECTORS
from src.data.feature_engineering import create_features, get_feature_columns
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.trainer import train_all_models


def run_verification():
    """Run the full pipeline across sectors and horizons."""
    
    # Load data once
    data_path = ROOT / ".." / "Dataset" / "USA ENGERY PREDICTION.xlsx"
    if not data_path.exists():
        # Try alternate location
        data_path = ROOT / "Data" / "USA ENGERY PREDICTION.xlsx"
    
    print(f"Loading data from: {data_path}")
    df = load_raw_data(str(data_path))
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    
    validation = validate_data(df)
    print(f"  Validation: {validation['n_rows']} rows, {validation['duplicates']} duplicates")
    
    # Test matrix
    sectors = ALL_SECTORS
    test_results = []
    total = 0
    passed = 0
    failed = 0
    
    for sector in sectors:
        print(f"\n{'='*60}")
        print(f"SECTOR: {sector}")
        print(f"{'='*60}")
        
        try:
            # Feature engineering
            features_df = create_features(df, sector)
            feat_cols = get_feature_columns(features_df)
            print(f"  Features: {len(feat_cols)} columns, {len(features_df)} samples")
            
            # Split & scale
            preproc = TimeSeriesPreprocessor()
            data = preproc.split(features_df, test_fraction=0.20)
            data = preproc.fit_transform(data)
            print(f"  Split: {len(data['X_train'])} train / {len(data['X_test'])} test")
            
            # Train all models
            results_df, best_models = train_all_models(
                data["X_train"], data["y_train"],
                data["X_test"], data["y_test"],
                data["y_train_orig"], data["y_test_orig"],
                preproc.scaler_y,
            )
            
            # Report
            best = results_df.iloc[0]
            print(f"\n  BEST MODEL: {best['Model']}")
            print(f"    RMSE = {best['RMSE']:.2f}")
            print(f"    R²   = {best['R2']:.4f}")
            print(f"    MAE  = {best['MAE']:.2f}")
            
            total += 1
            passed += 1
            test_results.append({
                "sector": sector,
                "status": "PASS",
                "best_model": best["Model"],
                "rmse": best["RMSE"],
                "r2": best["R2"],
            })
            
        except Exception as e:
            total += 1
            failed += 1
            test_results.append({
                "sector": sector,
                "status": "FAIL",
                "error": str(e),
            })
            print(f"  FAILED: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()
    for r in test_results:
        if r["status"] == "PASS":
            print(f"  ✓ {r['sector']:20s} -> {r['best_model']:20s} RMSE={r['rmse']:.2f}  R²={r['r2']:.4f}")
        else:
            print(f"  ✗ {r['sector']:20s} -> ERROR: {r['error']}")
    
    return passed == total


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
