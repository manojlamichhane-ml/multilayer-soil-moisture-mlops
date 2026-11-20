from config import load_params

P = load_params()
DEPTH_COLS = {"S30", "S60", "S90", "S120", "S150", "S180"}


def test_no_measured_soil_moisture_used_as_input():
    """Measured moisture at any depth would be leakage; only PREDICTED upper layers may be used."""
    for d, feats in P["features"].items():
        assert not DEPTH_COLS & set(feats), f"{d} uses a measured soil-moisture column"


def test_every_model_has_hyperparameters():
    for i, d in enumerate(P["depths"]):
        names = [d] if i == 0 else [f"{d}_indep", f"{d}_chain"]
        for n in names:
            assert n in P["hyperparameters"], n


def test_chain_mode_is_known():
    assert P["chain_mode"] in ("strict", "in_sample")
