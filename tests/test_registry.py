import json

from config import MODELS_DIR, load_params

P = load_params()
R = json.loads((MODELS_DIR / "registry.json").read_text())


def test_all_model_files_exist():
    for kind in ("independent", "chained"):
        for d in P["depths"]:
            assert (MODELS_DIR / kind / f"{d}.json").exists()


def test_every_depth_passes_quality_gate():
    assert all(R["quality_gate"]["passes"].values())


def test_presets_cover_every_input():
    for p in R["presets"].values():
        assert set(p["values"]) == set(R["inputs"])


def test_presets_stay_inside_training_range():
    for p in R["presets"].values():
        for f, v in p["values"].items():
            lo, hi = R["input_meta"][f]["range"]
            assert lo - 1e-6 <= v <= hi + 1e-6, f
