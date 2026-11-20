from fastapi.testclient import TestClient

from api import REG, app

client = TestClient(app)
TYPICAL = REG["presets"]["Typical day"]["values"]
SOIL = ("Sand", "Silt", "Clay", "OM", "Carbon", "pH")


def post(**kw):
    return client.post("/predict", json={"inputs": TYPICAL, **kw})


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_web_page():
    r = client.get("/")
    assert r.status_code == 200 and "Multilayer soil moisture" in r.text


def test_profile_has_six_layers_top_down():
    r = post()
    assert r.status_code == 200
    layers = r.json()["layers"]
    assert [L["depth_cm"] for L in layers] == [30, 60, 90, 120, 150, 180]
    for L in layers:
        assert 0 < L["value"] < 0.6


def test_all_four_models_returned_for_comparison():
    out = post().json()
    assert set(out["compare"]) == {m["key"] for m in REG["models"]}


def test_chained_model_feeds_the_layer_above():
    layers = post(model="with_soil_chained").json()["layers"]
    assert layers[0]["fed_from_above"] is None
    for above, below in zip(layers, layers[1:]):
        assert below["fed_from_above"] == above["value"]


def test_independent_model_is_not_fed():
    layers = post(model="no_soil_independent").json()["layers"]
    assert all(L["fed_from_above"] is None and L["upper_layer_effect"] is None for L in layers)


def test_no_soil_model_works_without_soil_inputs():
    inputs = {f: v for f, v in TYPICAL.items() if not f.startswith(SOIL)}
    r = client.post("/predict", json={"inputs": inputs, "model": "no_soil_chained"})
    assert r.status_code == 200
    assert set(r.json()["compare"]) == {"no_soil_chained", "no_soil_independent"}


def test_soil_model_needs_soil_inputs():
    inputs = {f: v for f, v in TYPICAL.items() if not f.startswith(SOIL)}
    assert client.post("/predict", json={"inputs": inputs, "model": "with_soil_chained"}).status_code == 422


def test_unknown_model_rejected():
    assert post(model="magic").status_code == 422


def test_impossible_ph_rejected():
    assert client.post("/predict", json={"inputs": {**TYPICAL, "pH (0-15 cm)": 20}}).status_code == 422


def test_out_of_range_gives_warning():
    hi = REG["input_meta"]["Elevation"]["range"][1]
    r = client.post("/predict", json={"inputs": {**TYPICAL, "Elevation": hi + 50}})
    assert r.status_code == 200 and r.json()["warnings"]
