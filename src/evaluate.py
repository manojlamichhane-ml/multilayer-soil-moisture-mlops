"""Stage 2 - evaluate the chain the way the paper does, and package it for the web app.

Validation schemes (each reported for independent AND chained models):
  holdout : the 25% test rows (shared by every depth)
  cv      : 5-fold cross-validation on the training rows (chain refit per fold)
  year    : train on 2020 + 2022, test on 2021 (leave-one-year-out)
  fields  : leave-one-field-out, pooled over all 18 fields

Outputs:
  models/registry.json       everything the app needs (inputs, ranges, presets, scores, drivers)
  reports/metrics.json       summary for `dvc metrics show`
  reports/validation.csv     every depth x model type x scheme
  reports/upper_layer_gain.csv
  reports/figures/*.png
Every scheme is logged to MLflow.
"""
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from chain import Chain, fit_chain  # noqa: E402
from config import (FIG_DIR, MLFLOW_URI, MODELS_DIR, REPORTS_DIR, UPPER_COL,  # noqa: E402
                    all_features, load_data, load_params, split_data)
from metrics import compute_metrics  # noqa: E402

KINDS = ("independent", "chained")

# ---------------------------------------------------------------------------
# Feature descriptions for the web app
# ---------------------------------------------------------------------------
def feature_meta(name):
    if name.startswith(("Sand", "Silt", "Clay")):
        return {"group": "Soil", "unit": "%", "limits": [0, 100]}
    if name.startswith(("OM", "Carbon")):
        return {"group": "Soil", "unit": "%", "limits": [0, 100]}
    if name.startswith("pH"):
        return {"group": "Soil", "unit": "", "limits": [0, 14]}
    if name == "Elevation":
        return {"group": "Terrain", "unit": "m", "limits": [0, 9000]}
    table = {
        "Blue": ("PlanetScope", "", [0, 1]), "NIR": ("PlanetScope", "", [0, 1]),
        "VARI": ("PlanetScope", "", [-2, 2]), "NDVIgb": ("PlanetScope", "", [-1, 1]),
        "SIPI": ("PlanetScope", "", [-2, 3]),
        "10-day antecedent Pcp": ("Weather", "mm", [0, 1000]), "7-day antecedent Pcp": ("Weather", "mm", [0, 1000]),
        "Ts": ("Weather", "°C", [-40, 60]), "Tmax": ("Weather", "°C", [-40, 60]),
        "Tmin": ("Weather", "°C", [-50, 50]), "Rn": ("Weather", "MJ/m²/day", [-10, 50]),
        "ETo": ("Weather", "mm/day", [0, 25]),
    }
    g, u, lim = table[name]
    return {"group": g, "unit": u, "limits": lim}


LABELS = {"Ts": "Soil temperature", "Tmax": "Max air temperature", "Tmin": "Min air temperature",
          "Rn": "Net radiation", "ETo": "Reference ET", "Blue": "Blue reflectance", "NIR": "NIR reflectance",
          "10-day antecedent Pcp": "Rain, last 10 days", "7-day antecedent Pcp": "Rain, last 7 days"}


# ---------------------------------------------------------------------------
def score_all(true_df, pred_df, depths):
    out = {}
    for d in depths:
        for k in KINDS:
            out[(d, k)] = compute_metrics(true_df[d], pred_df[f"{d}_{k}"])
    return out


def cross_validate(train, params):
    preds = []
    folds = params["split"]["cv_folds"]
    from sklearn.model_selection import KFold
    for tr, va in KFold(n_splits=folds, shuffle=True, random_state=params["seed"]).split(train):
        chain = fit_chain(train.iloc[tr], params)
        preds.append(chain.predict(train.iloc[va][all_features(params)]))
    return pd.concat(preds).loc[train.index]


def holdout_group(df, params, mask_fn):
    """Fit on rows where mask is False, predict rows where it is True, for every group."""
    preds = []
    for test_mask in mask_fn(df):
        chain = fit_chain(df[~test_mask], params)
        preds.append(chain.predict(df[test_mask][all_features(params)]))
    return pd.concat(preds).loc[df.index.intersection(pd.concat(preds).index)]


def drivers(chain, train, params):
    """Share of each input in the chained model's predictions (mean |contribution|, %)."""
    depths, feats = params["depths"], params["features"]
    pred = chain.predict(train[all_features(params)])
    out = {}
    for i, d in enumerate(depths):
        X = train[feats[d]]
        if i > 0:
            X = X.assign(**{UPPER_COL: pred[f"{depths[i - 1]}_chained"].values})
        contrib = chain.chained[d].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        share = np.abs(contrib).mean(axis=0)
        share = share / share.sum() * 100
        out[d] = sorted([{"feature": f, "share_pct": round(float(s), 1)} for f, s in zip(X.columns, share)],
                        key=lambda r: -r["share_pct"])
    return out


def presets(df, params):
    """Typical input values for a few situations (medians of many rows; no raw data exposed)."""
    feats = all_features(params)
    static = [f for f in feats if feature_meta(f)["group"] in ("Soil", "Terrain")]
    base = df[feats].median()
    rain = df["10-day antecedent Pcp"]
    subsets = {
        "Typical day": df,
        "After heavy rain": df[rain >= rain.quantile(0.9)],
        "Hot, dry spell": df[(rain <= rain.quantile(0.25)) & (df["Tmax"] >= df["Tmax"].quantile(0.75))],
        "Cool season": df[df["Ts"] <= df["Ts"].quantile(0.2)],
    }
    out = {}
    for name, sub in subsets.items():
        vals = sub[feats].median()
        vals[static] = base[static]
        out[name] = {"values": {f: round(float(v), 4) for f, v in vals.items()}, "n_rows": int(len(sub))}
    return out


# ---------------------------------------------------------------------------
def figures(val, gain, drv, params):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    depths = params["depths"]
    cm = [int(d[1:]) for d in depths]
    IND, CHN, INK, RULE = "#B08D57", "#1F78A8", "#1F2420", "#D9DCD6"

    # 1. hold-out R² by depth
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    h = val[val.scheme == "holdout"]
    x = np.arange(len(depths))
    for off, kind, col, lab in [(-0.2, "independent", IND, "Independent"), (0.2, "chained", CHN, "With layer above")]:
        v = [h[(h.depth == d) & (h.kind == kind)].r2.iloc[0] for d in depths]
        bars = ax.bar(x + off, v, 0.38, color=col, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{c} cm" for c in cm]); ax.set_ylim(0, 1.05); ax.set_ylabel("Hold-out R²")
    ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color=RULE); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Hold-out R² at each depth", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "r2_by_depth.png", dpi=180); plt.close(fig)

    # 2. R² vs depth for every validation scheme (chained), as a soil profile
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    styles = {"holdout": ("Hold-out test", "#1F78A8", "o"), "cv": ("Cross-validation", "#5AAE61", "s"),
              "year": ("Unseen year (2021)", "#C2185B", "^"), "fields": ("Unseen fields", "#8C6D46", "D")}
    for sch, (lab, col, mk) in styles.items():
        s = val[(val.scheme == sch) & (val.kind == "chained")].set_index("depth").loc[depths]
        ax.plot(s.r2, cm, marker=mk, color=col, lw=2, ms=6, label=lab)
    ax.invert_yaxis(); ax.set_yticks(cm); ax.set_ylabel("Depth (cm)"); ax.set_xlabel("R² (chained models)")
    ax.set_xlim(min(0, val.r2.min() - 0.05), 1)
    ax.spines[["top", "right"]].set_visible(False); ax.xaxis.grid(True, color=RULE); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("How well the chain transfers", loc="left", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "validation_profile.png", dpi=180); plt.close(fig)

    # 3. drivers heatmap (share of prediction, chained models)
    groups = {"Upper layer": [UPPER_COL]}
    for f in all_features(params):
        groups.setdefault(feature_meta(f)["group"], []).append(f)
    gnames = ["Upper layer", "PlanetScope", "Weather", "Soil", "Terrain"]
    mat = np.array([[sum(r["share_pct"] for r in drv[d] if r["feature"] in groups[g]) for g in gnames] for d in depths])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(60, mat.max()))
    ax.set_xticks(range(len(gnames)), gnames, rotation=20, ha="right"); ax.set_yticks(range(len(depths)), [f"{c} cm" for c in cm])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 35 else INK)
    ax.set_title("What drives each layer (% of prediction)", loc="left", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(FIG_DIR / "drivers_by_depth.png", dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params()
    depths = params["depths"]
    df = load_data(params)
    train, test = split_data(df, params)
    feats = all_features(params)
    chain = Chain.load(MODELS_DIR, depths, params["features"])

    print("Hold-out ...")
    results = {"holdout": score_all(test, chain.predict(test[feats]), depths)}
    alt_mode = "in_sample" if params["chain_mode"] == "strict" else "strict"
    alt = fit_chain(train, params, mode=alt_mode)
    results[f"holdout_{alt_mode}"] = score_all(test, alt.predict(test[feats]), depths)
    print("Cross-validation ...")
    results["cv"] = score_all(train, cross_validate(train, params), depths)
    print("Unseen year ...")
    yr = params["evaluation"]["test_year"]
    results["year"] = score_all(df[df.Year == yr], holdout_group(df, params, lambda d: [d.Year == yr]), depths)
    if params["evaluation"]["run_field_holdout"]:
        print("Unseen fields (18 refits) ...")
        p = holdout_group(df, params, lambda d: [d.Field == f for f in d.Field.unique()])
        results["fields"] = score_all(df.loc[p.index], p, depths)

    rows = [{"scheme": s, "depth": d, "kind": k, **m} for s, res in results.items() for (d, k), m in res.items()]
    val = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    val.to_csv(REPORTS_DIR / "validation.csv", index=False)

    gain = []
    for d in depths[1:]:
        a = results["holdout"][(d, "independent")]
        b = results["holdout"][(d, "chained")]
        gain.append({"depth": d, "rmse_reduction_pct": round((a["rmse"] - b["rmse"]) / a["rmse"] * 100, 1),
                     "r2_change": round(b["r2"] - a["r2"], 3)})
    gain = pd.DataFrame(gain)
    gain.to_csv(REPORTS_DIR / "upper_layer_gain.csv", index=False)

    drv = drivers(chain, train, params)
    gate = params["quality_gate"]["min_test_r2"]
    passes = {d: results["holdout"][(d, "chained")]["r2"] >= gate for d in depths}

    # ---- MLflow -------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("multilayer-soil-moisture")
    with mlflow.start_run(run_name=f"chain-{params['chain_mode']}-{datetime.now():%Y%m%d-%H%M}"):
        mlflow.log_params({"chain_mode": params["chain_mode"], "test_size": params["split"]["test_size"],
                           "split_seed": params["split"]["random_state"], "n_train": len(train), "n_test": len(test)})
        for (s, d, k), m in val.set_index(["scheme", "depth", "kind"]).iterrows():
            for metric in ("r2", "rmse", "rrmse_pct"):
                mlflow.log_metric(f"{s}.{d}.{k}.{metric}", m[metric])
        mlflow.set_tag("all_depths_pass_gate", all(passes.values()))

    # ---- summary + registry --------------------------------------------------
    summary = {"chain_mode": params["chain_mode"],
               "holdout_chained_r2": {d: results["holdout"][(d, "chained")]["r2"] for d in depths},
               "holdout_chained_rmse": {d: results["holdout"][(d, "chained")]["rmse"] for d in depths},
               "holdout_independent_r2": {d: results["holdout"][(d, "independent")]["r2"] for d in depths},
               "all_depths_pass_gate": all(passes.values())}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    nested = {}
    for scheme, res in results.items():
        for (d, k), m in res.items():
            nested.setdefault(d, {}).setdefault(k, {})[scheme] = m
    registry = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "chain_mode": params["chain_mode"], "depths": depths, "upper_col": UPPER_COL,
        "features": params["features"], "inputs": feats,
        "input_meta": {f: {**feature_meta(f), "label": LABELS.get(f, f),
                           "range": [round(float(train[f].min()), 4), round(float(train[f].max()), 4)],
                           "default": round(float(train[f].median()), 4)} for f in feats},
        "presets": presets(df, params),
        "observed_range": {d: [round(float(df[d].min()), 3), round(float(df[d].max()), 3)] for d in depths},
        "metrics": nested, "gain": gain.to_dict(orient="records"), "drivers": drv,
        "quality_gate": {"min_test_r2": gate, "passes": passes},
        "n_rows": {"total": len(df), "train": len(train), "test": len(test), "fields": int(df.Field.nunique()),
                   "years": sorted(int(y) for y in df.Year.unique())},
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2))

    figures(val[val.scheme.isin(["holdout", "cv", "year", "fields"])], gain, drv, params)

    print("\nChained hold-out R²: " + "  ".join(f"{d[1:]}cm {results['holdout'][(d, 'chained')]['r2']:.3f}" for d in depths))
    print("Independent       : " + "  ".join(f"{d[1:]}cm {results['holdout'][(d, 'independent')]['r2']:.3f}" for d in depths))
    print(f"Quality gate (R² ≥ {gate}): {'all depths pass' if all(passes.values()) else passes}")
