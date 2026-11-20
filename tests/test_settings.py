from config import MODEL_KEYS, load_params

P = load_params()
DEPTH_COLS = {"S30", "S60", "S90", "S120", "S150", "S180"}
SOIL = ("Sand", "Silt", "Clay", "OM", "Carbon", "pH")


def test_no_measured_soil_moisture_used_as_input():
    """Measured moisture at any depth would be leakage; only PREDICTED upper layers may be used."""
    for v in P["variants"]:
        for d, feats in P["features"][v].items():
            assert not DEPTH_COLS & set(feats), f"{v} {d} uses a measured soil-moisture column"


def test_no_soil_variant_has_no_soil_properties():
    for d, feats in P["features"]["no_soil"].items():
        assert not [f for f in feats if f.startswith(SOIL)], d


def test_every_variant_covers_every_depth():
    for v in P["variants"]:
        assert set(P["features"][v]) == set(P["depths"]), v


def test_four_models_defined():
    assert [k for k, _, _ in MODEL_KEYS] == ["with_soil_chained", "with_soil_independent",
                                            "no_soil_chained", "no_soil_independent"]


def test_every_model_has_hyperparameters():
    for i, d in enumerate(P["depths"]):
        names = [d] if i == 0 else [f"{d}_indep", f"{d}_chain"]
        for n in names:
            assert n in P["hyperparameters"], n


def test_chain_mode_is_known():
    assert P["chain_mode"] in ("strict", "in_sample")
