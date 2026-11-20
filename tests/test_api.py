from fastapi.testclient import TestClient

from api import REG, app

client = TestClient(app)
TYPICAL = REG["presets"]["Typical day"]["values"]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_web_page():
    r = client.get("/")
    assert r.status_code == 200 and "layer by layer" in r.text


def test_profile_has_six_layers_top_down():
    r = client.post("/predict", json={"inputs": TYPICAL})
    assert r.status_code == 200
    layers = r.json()["layers"]
    assert [L["depth_cm"] for L in layers] == [30, 60, 90, 120, 150, 180]
    for L in layers:
        assert 0 < L["chained"] < 0.6 and 0 < L["independent"] < 0.6


def test_each_layer_is_fed_the_layer_above():
    layers = client.post("/predict", json={"inputs": TYPICAL}).json()["layers"]
    assert layers[0]["fed_from_above"] is None
    for above, below in zip(layers, layers[1:]):
        assert below["fed_from_above"] == above["chained"]


def test_missing_input_rejected():
    inputs = dict(TYPICAL)
    inputs.pop(next(iter(inputs)))
    assert client.post("/predict", json={"inputs": inputs}).status_code == 422


def test_impossible_ph_rejected():
    assert client.post("/predict", json={"inputs": {**TYPICAL, "pH (0-15 cm)": 20}}).status_code == 422


def test_out_of_range_gives_warning():
    hi = REG["input_meta"]["Elevation"]["range"][1]
    r = client.post("/predict", json={"inputs": {**TYPICAL, "Elevation": hi + 50}})
    assert r.status_code == 200 and r.json()["warnings"]
