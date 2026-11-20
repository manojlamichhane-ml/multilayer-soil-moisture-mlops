"""Stage 1 - train: fit the six-depth chain (independent + chained models) on the training rows.

Outputs:
  models/independent/<depth>.json, models/chained/<depth>.json   (XGBoost native format)
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
    chain = fit_chain(train, params)
    chain.save(MODELS_DIR)

    info = {"chain_mode": params["chain_mode"], "n_rows": len(df), "n_train": len(train), "n_test": len(test),
            "n_train_180": int(train["S180"].notna().sum()), "n_test_180": int(test["S180"].notna().sum()),
            "seconds": round(time.time() - start, 1)}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"Trained 11 models ({params['chain_mode']} chain) on {len(train)} rows in {info['seconds']} s "
          f"-> {MODELS_DIR}")
