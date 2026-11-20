"""Stage 2 - evaluate the four models the way the paper does, and package them for the web app.

The four models: {with, without} soil properties  x  {with, without} the predicted layer above.

Validation schemes (each reported for all four models):
  holdout : the 25% test rows (shared by every depth)
  cv      : 5-fold cross-validation on the training rows (chain refit per fold)
  year    : train on 2020 + 2022, test on 2021 (leave-one-year-out)
  fields  : leave-one-field-out, pooled over all 18 fields

Outputs:
  models/registry.json       everything the app needs (inputs, ranges, presets, scores, drivers)
  reports/metrics.json       summary for `dvc metrics show`
  reports/validation.csv     every model x depth x scheme
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

from chain import Chain, fit_chain, fit_paper  # noqa: E402
from config import (FIG_DIR, MLFLOW_URI, MODEL_KEYS, MODELS_DIR, REPORTS_DIR, UPPER_COL,  # noqa: E402
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
        return {"group": "Topography", "unit": "m", "limits": [0, 9000]}
    table = {
        "Blue": ("Satellite", "", [0, 1]), "NIR": ("Satellite", "", [0, 1]),
        "VARI": ("Satellite", "", [-2, 2]), "NDVIgb": ("Satellite", "", [-1, 1]),
        "SIPI": ("Satellite", "", [-2, 3]),
        "10-day antecedent Pcp": ("Climate", "mm", [0, 1000]), "7-day antecedent Pcp": ("Climate", "mm", [0, 1000]),
        "Ts": ("Climate", "°C", [-40, 60]), "Tmax": ("Climate", "°C", [-40, 60]),
        "Tmin": ("Climate", "°C", [-50, 50]), "Rn": ("Climate", "MJ/m²/day", [-10, 50]),
        "ETo": ("Climate", "mm/day", [0, 25]),
    }
    g, u, lim = table[name]
    return {"group": g, "unit": u, "limits": lim}


LABELS = {"Ts": "Soil temperature", "Tmax": "Max air temperature", "Tmin": "Min air temperature",
          "Rn": "Net radiation", "ETo": "Reference ET", "Blue": "Blue reflectance", "NIR": "NIR reflectance",
          "10-day antecedent Pcp": "Rain, last 10 days", "7-day antecedent Pcp": "Rain, last 7 days",
          "Elevation": "Elevation"}

# How the four models are named in the app and the figures
MODEL_INFO = {
    "with_soil_chained": {"label": "With soil properties and upper-layer SM", "short": "Soil + chain",
                          "note": "The model in the paper"},
    "with_soil_independent": {"label": "With soil properties, without upper-layer SM", "short": "Soil, no chain",
                              "note": "Every depth predicted on its own"},
    "no_soil_chained": {"label": "Without soil properties, with upper-layer SM", "short": "Chain, no soil",
                        "note": "Satellite, climate and elevation, plus the layer above"},
    "no_soil_independent": {"label": "Without soil properties and upper-layer SM", "short": "No soil, no chain",
                            "note": "Satellite, climate and elevation only"},
}
# Hue = soil properties (orange used, blue not used); hatching = each layer predicted alone
VARIANT_COLORS = {"with_soil": "#C4621B", "no_soil": "#1F78B4"}
GROUPS = ["Upper layer", "Satellite", "Climate", "Soil", "Topography"]


# ---------------------------------------------------------------------------
def score_all(true_df, pred_df, depths):
    """{(depth, kind): metrics} for one variant's predictions."""
    return {(d, k): compute_metrics(true_df[d], pred_df[f"{d}_{k}"]) for d in depths for k in KINDS}


def cross_validate(train, params, variant):
    from sklearn.model_selection import KFold
    preds = []
    for tr, va in KFold(n_splits=params["split"]["cv_folds"], shuffle=True, random_state=params["seed"]).split(train):
        chain = fit_chain(train.iloc[tr], params, variant)
        preds.append(chain.predict(train.iloc[va]))
    return pd.concat(preds).loc[train.index]


def holdout_group(df, params, variant, masks):
    """For every mask: fit on rows where it is False, predict rows where it is True."""
    preds = [fit_chain(df[~m], params, variant).predict(df[m]) for m in masks]
    return pd.concat(preds)


def drivers(chain, train, params, kind):
    """Share of each input in one model's predictions (mean |contribution|, %), per depth."""
    depths = params["depths"]
    pred = chain.predict(train)
    out = {}
    for i, d in enumerate(depths):
        X = train[chain.features[d]]
        model = getattr(chain, kind)[d]
        if kind == "chained" and i > 0:
            X = X.assign(**{UPPER_COL: pred[f"{depths[i - 1]}_chained"].values})
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        share = np.abs(contrib).mean(axis=0)
        share = share / share.sum() * 100
        out[d] = sorted([{"feature": f, "share_pct": round(float(s), 1)} for f, s in zip(X.columns, share)],
                        key=lambda r: -r["share_pct"])
    return out


def group_of(feature):
    return "Upper layer" if feature == UPPER_COL else feature_meta(feature)["group"]


def presets(df, params):
    """Typical input values for a few situations (medians of many rows; no raw data exposed)."""
    feats = all_features(params)
    static = [f for f in feats if feature_meta(f)["group"] in ("Soil", "Topography")]
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
def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    getattr(ax, f"{grid}axis").grid(True, color="#D9DCD6")
    ax.set_axisbelow(True)


def figures(val, drv, params):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    depths = params["depths"]
    cm = [int(d[1:]) for d in depths]
    INK = "#1F2420"
    h = val[val.scheme == "holdout"]

    # 1. hold-out R² by depth, four models
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x, w = np.arange(len(depths)), 0.2
    for j, (key, _, _) in enumerate(MODEL_KEYS):
        v = [h[(h.model == key) & (h.depth == d)].r2.iloc[0] for d in depths]
        col = VARIANT_COLORS[key.rsplit("_", 1)[0]]
        hatched = key.endswith("independent")
        ax.bar(x + (j - 1.5) * w, v, w * 0.92, facecolor="white" if hatched else col, edgecolor=col,
               hatch="////" if hatched else None, linewidth=1.2, label=MODEL_INFO[key]["label"])
    ax.set_xticks(x, [f"{c} cm" for c in cm]); ax.set_ylim(min(0, h.r2.min() - 0.05), 1.0); ax.set_ylabel("Hold-out R²")
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    ax.set_title("Hold-out R² at each depth, four models", loc="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(); fig.savefig(FIG_DIR / "r2_by_depth.png", dpi=180); plt.close(fig)

    # 2. R² vs depth for every validation scheme, paper model (solid) vs no-soil chain (dashed)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    styles = {"holdout": ("Hold-out test", "#1F78A8", "o"), "cv": ("Cross-validation", "#5AAE61", "s"),
              "year": ("Unseen year (2021)", "#C2185B", "^"), "fields": ("Unseen fields", "#8C6D46", "D")}
    for sch, (lab, col, mk) in styles.items():
        for key, ls in (("with_soil_chained", "-"), ("no_soil_chained", ":")):
            s = val[(val.scheme == sch) & (val.model == key)].set_index("depth").loc[depths]
            ax.plot(s.r2, cm, marker=mk, color=col, lw=2, ms=5.5, ls=ls,
                    label=lab if ls == "-" else None, alpha=1 if ls == "-" else 0.75)
    ax.plot([], [], color="#777", ls="-", label="With soil properties and upper-layer SM")
    ax.plot([], [], color="#777", ls=":", label="Without soil properties, with upper-layer SM")
    ax.invert_yaxis(); ax.set_yticks(cm); ax.set_ylabel("Depth (cm)"); ax.set_xlabel("R²")
    ax.set_xlim(min(0, val.r2.min() - 0.05), 1)
    style(ax, "x")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("How well the chain transfers", loc="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(); fig.savefig(FIG_DIR / "validation_profile.png", dpi=180); plt.close(fig)

    # 3. drivers heatmap, paper model
    d0 = drv["with_soil_chained"]
    mat = np.array([[sum(r["share_pct"] for r in d0[d] if group_of(r["feature"]) == g) for g in GROUPS]
                    for d in depths])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(60, mat.max()))
    ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=20, ha="right")
    ax.set_yticks(range(len(depths)), [f"{c} cm" for c in cm])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 35 else INK)
    ax.set_title("What drives each layer (% of prediction, paper model)", loc="left", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(FIG_DIR / "drivers_by_depth.png", dpi=180); plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params()
    depths = params["depths"]
    df = load_data(params)
    train, test = split_data(df, params)
    yr = params["evaluation"]["test_year"]
    paper = params["chain_mode"] == "paper"

    rows, drv, chains = [], {}, {}
    for variant in params["variants"]:
        print(f"\n=== {variant} ===")
        chain = Chain.load(MODELS_DIR / variant, depths, params["features"][variant])
        chains[variant] = chain
        if paper:
            # the published protocol: each model scored on its own split (same seeds as training)
            _, ho, cvm = fit_paper(df, params, variant)
            res = {"holdout": ho, "cv": cvm}
            strict = fit_chain(train, params, variant, mode="strict")
            res["holdout_strict"] = score_all(test, strict.predict(test), depths)   # leak-free reference
        else:
            res = {"holdout": score_all(test, chain.predict(test), depths)}
            alt = fit_chain(train, params, variant, mode="in_sample")
            res["holdout_in_sample"] = score_all(test, alt.predict(test), depths)
            print("Cross-validation ...")
            res["cv"] = score_all(train, cross_validate(train, params, variant), depths)
        print("Unseen year ...")
        p = holdout_group(df, params, variant, [df.Year == yr])
        res["year"] = score_all(df.loc[p.index], p, depths)
        if params["evaluation"]["run_field_holdout"]:
            print("Unseen fields (18 refits) ...")
            p = holdout_group(df, params, variant, [df.Field == f for f in df.Field.unique()])
            res["fields"] = score_all(df.loc[p.index], p, depths)
        for scheme, r in res.items():
            for (d, k), m in r.items():
                rows.append({"model": f"{variant}_{k}", "variant": variant, "kind": k, "scheme": scheme,
                             "depth": d, **m})
        for k in KINDS:
            drv[f"{variant}_{k}"] = drivers(chain, train, params, k)

    val = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    val.to_csv(REPORTS_DIR / "validation.csv", index=False)

    def r(key, scheme, d, metric="r2"):
        s = val[(val.model == key) & (val.scheme == scheme) & (val.depth == d)]
        return float(s[metric].iloc[0])

    gain = []
    for variant in params["variants"]:
        for d in depths[1:]:
            a, b = r(f"{variant}_independent", "holdout", d, "rmse"), r(f"{variant}_chained", "holdout", d, "rmse")
            gain.append({"variant": variant, "depth": d, "rmse_reduction_pct": round((a - b) / a * 100, 1),
                         "r2_change": round(r(f"{variant}_chained", "holdout", d) -
                                            r(f"{variant}_independent", "holdout", d), 3)})
    gain = pd.DataFrame(gain)
    gain.to_csv(REPORTS_DIR / "upper_layer_gain.csv", index=False)

    gate = params["quality_gate"]["min_test_r2"]
    passes = {key: {d: r(key, "holdout", d) >= gate for d in depths} for key, _, _ in MODEL_KEYS}
    paper_ok = all(passes["with_soil_chained"].values())

    # ---- MLflow -------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("multilayer-soil-moisture")
    with mlflow.start_run(run_name=f"four-models-{params['chain_mode']}-{datetime.now():%Y%m%d-%H%M}"):
        mlflow.log_params({"chain_mode": params["chain_mode"], "variants": ",".join(params["variants"]),
                           "test_size": params["split"]["test_size"], "split_seed": params["split"]["random_state"],
                           "n_train": len(train), "n_test": len(test)})
        for _, m in val.iterrows():
            for metric in ("r2", "rmse", "rrmse_pct"):
                mlflow.log_metric(f"{m.model}.{m.scheme}.{m.depth}.{metric}", m[metric])
        mlflow.set_tag("paper_model_passes_gate", paper_ok)

    # ---- summary + registry --------------------------------------------------
    summary = {"chain_mode": params["chain_mode"],
               "holdout_r2": {key: {d: r(key, "holdout", d) for d in depths} for key, _, _ in MODEL_KEYS},
               "holdout_rmse": {key: {d: r(key, "holdout", d, "rmse") for d in depths} for key, _, _ in MODEL_KEYS},
               "paper_model_passes_gate": paper_ok}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    metrics = {}
    for _, m in val.iterrows():
        metrics.setdefault(m.model, {}).setdefault(m.depth, {})[m.scheme] = {
            "r2": m.r2, "rmse": m.rmse, "rrmse_pct": m.rrmse_pct, "n": int(m.n)}
    feats = all_features(params)
    registry = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "chain_mode": params["chain_mode"], "depths": depths, "upper_col": UPPER_COL,
        "variants": params["variants"],
        "models": [{"key": key, "variant": v, "kind": k, **MODEL_INFO[key], "color": VARIANT_COLORS[v],
                    "uses_soil": v == "with_soil", "uses_upper_layer": k == "chained"} for key, v, k in MODEL_KEYS],
        "default_model": "with_soil_chained",
        "features": params["features"], "inputs": feats,
        "input_meta": {f: {**feature_meta(f), "label": LABELS.get(f, f),
                           "range": [round(float(train[f].min()), 4), round(float(train[f].max()), 4)],
                           "default": round(float(train[f].median()), 4)} for f in feats},
        "presets": presets(df, params),
        "observed_range": {d: [round(float(df[d].min()), 3), round(float(df[d].max()), 3)] for d in depths},
        "metrics": metrics, "gain": gain.to_dict(orient="records"), "drivers": drv, "groups": GROUPS,
        "quality_gate": {"min_test_r2": gate, "model": "with_soil_chained", "passes": passes},
        "n_rows": {"total": len(df), "train": len(train), "test": len(test), "fields": int(df.Field.nunique()),
                   "years": sorted(int(y) for y in df.Year.unique())},
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2))

    figures(val[val.scheme.isin(["holdout", "cv", "year", "fields"])], drv, params)

    print("\nHold-out R²")
    for key, _, _ in MODEL_KEYS:
        print(f"  {MODEL_INFO[key]['label']:<34}" + "  ".join(f"{d[1:]:>3}cm {r(key, 'holdout', d):.3f}" for d in depths))
    print(f"Quality gate (paper model, R² ≥ {gate}): {'all depths pass' if paper_ok else passes['with_soil_chained']}")
