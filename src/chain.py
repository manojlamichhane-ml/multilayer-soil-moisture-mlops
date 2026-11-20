"""The layer-by-layer model chain from the paper.

Two versions of the chain are trained (see `variants` in params.yaml): with and
without soil properties. In each version, every depth has two XGBoost models:
  independent : uses only that depth's own inputs
  chained     : uses the same inputs PLUS the predicted soil moisture of the
                layer directly above (30 → 60 → 90 → 120 → 150 → 180 cm)

At prediction time the chain runs top-down: the 30 cm prediction is fed to the
60 cm model, its output to the 90 cm model, and so on.

During training, the "layer above" input for each training row must be a value
the upper model could have produced WITHOUT having seen that row; otherwise the
next model learns from unrealistically accurate inputs. In `strict` mode this
is done with out-of-fold (OOF) predictions. `in_sample` / `paper` modes reproduce
the published setup, where the upper model predicts its own training rows.

`fit_paper` reproduces the published notebook exactly: every model gets its own
75/25 split, and its hold-out and cross-validation scores come from that split.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from config import UPPER_COL
from metrics import compute_metrics


def xgb_params(params, name):
    n, depth, sub, col, mcw = params["hyperparameters"][name]
    return dict(objective="reg:squarederror", n_estimators=n, max_depth=depth, subsample=sub,
                colsample_bytree=col, gamma=0.0, min_child_weight=mcw,
                learning_rate=params["learning_rate"], random_state=params["seed"], n_jobs=4)


def model_name(depth, kind, first_depth):
    return depth if depth == first_depth else f"{depth}_{'chain' if kind == 'chained' else 'indep'}"


@dataclass
class Chain:
    depths: list
    features: dict
    independent: dict = field(default_factory=dict)
    chained: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- predict
    def predict(self, X):
        """Return a DataFrame with columns '<depth>_independent' and '<depth>_chained'."""
        out, feed = {}, None
        for i, d in enumerate(self.depths):
            Xd = X[self.features[d]]
            out[f"{d}_independent"] = self.independent[d].predict(Xd)
            if i == 0:
                out[f"{d}_chained"] = out[f"{d}_independent"]
            else:
                out[f"{d}_chained"] = self.chained[d].predict(Xd.assign(**{UPPER_COL: feed}))
            feed = out[f"{d}_chained"]
        return pd.DataFrame(out, index=X.index)

    # ------------------------------------------------------------- save/load
    def save(self, folder):
        folder = Path(folder)
        for kind in ("independent", "chained"):
            (folder / kind).mkdir(parents=True, exist_ok=True)
            for d, m in getattr(self, kind).items():
                m.save_model(folder / kind / f"{d}.json")

    @classmethod
    def load(cls, folder, depths, features):
        folder, chain = Path(folder), cls(depths, features)
        for kind in ("independent", "chained"):
            for d in depths:
                m = xgb.XGBRegressor()
                m.load_model(folder / kind / f"{d}.json")
                getattr(chain, kind)[d] = m
        return chain


def _oof(params, name, X, y, folds, seed):
    """Out-of-fold predictions: each row is predicted by a model that never saw it."""
    pred = pd.Series(np.nan, index=X.index)
    for tr, va in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        m = xgb.XGBRegressor(**xgb_params(params, name)).fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[va] = m.predict(X.iloc[va])
    return pred


def fit_chain(train, params, variant="with_soil", mode=None):
    """Train the independent and chained models for all depths on `train`, using one variant's inputs."""
    mode = mode or params["chain_mode"]
    depths, feats = params["depths"], params["features"][variant]
    chain, feed = Chain(depths, feats), None
    folds, seed = params["split"]["cv_folds"], params["seed"]

    for i, d in enumerate(depths):
        X = train[feats[d]]
        rows = train[d].notna()            # 180 cm was measured on fewer dates
        y = train[d]

        ind = xgb.XGBRegressor(**xgb_params(params, model_name(d, "independent", depths[0])))
        chain.independent[d] = ind.fit(X[rows], y[rows])

        if i == 0:
            chain.chained[d] = chain.independent[d]
            Xc, name = X, d
        else:
            Xc, name = X.assign(**{UPPER_COL: feed}), model_name(d, "chained", depths[0])
            chain.chained[d] = xgb.XGBRegressor(**xgb_params(params, name)).fit(Xc[rows], y[rows])

        if i < len(depths) - 1:            # build the input for the next layer down
            if mode == "strict":
                feed = _oof(params, name, Xc, y, folds, seed)
            else:
                feed = pd.Series(chain.chained[d].predict(Xc), index=X.index)
    return chain


def _paper_seed(params, name):
    return params["split"].get("seeds", {}).get(name, params["split"]["random_state"])


def fit_paper(df, params, variant="with_soil"):
    """The published protocol (notebook 05), on all labelled rows.

    Returns the fitted chain, plus hold-out and 5-fold CV metrics for every model:
    {(depth, kind): metrics}. Each model is scored on its own 25% test split.
    """
    from sklearn.model_selection import train_test_split
    depths, feats = params["depths"], params["features"][variant]
    s = params["split"]
    chain, feed, holdout, cv = Chain(depths, feats), None, {}, {}

    def fit_one(X, y, name):
        rows = y.notna()
        Xtr, Xte, ytr, yte = train_test_split(X[rows], y[rows], test_size=s["test_size"],
                                              random_state=_paper_seed(params, name))
        folds = []
        for tr, va in KFold(n_splits=s["cv_folds"], shuffle=True, random_state=params["seed"]).split(Xtr):
            m = xgb.XGBRegressor(**xgb_params(params, name)).fit(Xtr.iloc[tr], ytr.iloc[tr])
            folds.append(compute_metrics(ytr.iloc[va], m.predict(Xtr.iloc[va])))
        cvm = {k: round(float(np.mean([f[k] for f in folds])), 4) for k in ("r2", "rmse", "rrmse_pct")}
        cvm["n"] = len(Xtr)
        model = xgb.XGBRegressor(**xgb_params(params, name)).fit(Xtr, ytr)
        return model, compute_metrics(yte, model.predict(Xte)), cvm

    for i, d in enumerate(depths):
        X, y = df[feats[d]], df[d]
        name = model_name(d, "independent", depths[0])
        chain.independent[d], holdout[(d, "independent")], cv[(d, "independent")] = fit_one(X, y, name)
        if i == 0:
            chain.chained[d] = chain.independent[d]
            holdout[(d, "chained")], cv[(d, "chained")] = holdout[(d, "independent")], cv[(d, "independent")]
            Xc = X
        else:
            Xc = X.assign(**{UPPER_COL: feed})
            name = model_name(d, "chained", depths[0])
            chain.chained[d], holdout[(d, "chained")], cv[(d, "chained")] = fit_one(Xc, y, name)
        feed = pd.Series(chain.chained[d].predict(Xc), index=X.index)   # all rows, as in the paper
    return chain, holdout, cv
