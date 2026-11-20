"""The chain logic on synthetic data (runs on GitHub, where the real data is not available)."""
import numpy as np
import pandas as pd

from chain import Chain, fit_chain
from config import UPPER_COL, all_features, load_params

P = load_params()


def synthetic(n=160, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(all_features(P)))), columns=all_features(P))
    level = rng.random(n)
    for i, d in enumerate(P["depths"]):
        df[d] = 0.1 + 0.2 * level + 0.01 * i + rng.normal(0, 0.005, n)
    df.loc[: n // 4, "S180"] = np.nan          # 180 cm measured less often, like the real data
    return df


def test_chain_predicts_every_depth_both_ways(tmp_path):
    df = synthetic()
    chain = fit_chain(df, P)
    out = chain.predict(df[all_features(P)])
    assert out.shape == (len(df), 2 * len(P["depths"]))
    assert out.notna().all().all()


def test_first_layer_is_identical_in_both_modes():
    df = synthetic()
    out = fit_chain(df, P).predict(df[all_features(P)])
    assert np.allclose(out["S30_independent"], out["S30_chained"])


def test_chained_models_use_the_layer_above():
    chain = fit_chain(synthetic(), P)
    for d in P["depths"][1:]:
        assert UPPER_COL in chain.chained[d].get_booster().feature_names


def test_save_and_load_give_same_predictions(tmp_path):
    df = synthetic()
    chain = fit_chain(df, P)
    chain.save(tmp_path)
    again = Chain.load(tmp_path, P["depths"], P["features"])
    X = df[all_features(P)]
    assert np.allclose(chain.predict(X).values, again.predict(X).values)
