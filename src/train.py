"""Stage 1 - train: fit both versions of the six-depth chain (with and without soil properties).

Outputs:
  models/<variant>/independent/<depth>.json, models/<variant>/chained/<depth>.json   (XGBoost native format)
  reports/train_info.json
"""
import json
import time

from chain import fit_chain
from config import MODELS_DIR, REPORTS_DIR, load_data, load_params, split_data

if __name__ == "__main__":
    params = load_params()
    df = load_data(params)
    train, test = split_data(df, params)
    start = time.time()
    for variant in params["variants"]:
        fit_chain(train, params, variant).save(MODELS_DIR / variant)
        print(f"  {variant}: 11 models saved to {MODELS_DIR / variant}")

    info = {"chain_mode": params["chain_mode"], "variants": params["variants"], "n_rows": len(df),
            "n_train": len(train), "n_test": len(test),
            "n_train_180": int(train["S180"].notna().sum()), "n_test_180": int(test["S180"].notna().sum()),
            "seconds": round(time.time() - start, 1)}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"Trained {11 * len(params['variants'])} models ({params['chain_mode']} chain) on {len(train)} rows "
          f"in {info['seconds']} s")
