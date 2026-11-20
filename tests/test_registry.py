import json

from config import MODEL_KEYS, MODELS_DIR, load_params

P = load_params()
R = json.loads((MODELS_DIR / "registry.json").read_text())


def test_all_model_files_exist():
    for v in P["variants"]:
        for kind in ("independent", "chained"):
            for d in P["depths"]:
                assert (MODELS_DIR / v / kind / f"{d}.json").exists()


def test_paper_model_passes_quality_gate():
    assert all(R["quality_gate"]["passes"]["with_soil_chained"].values())


def test_registry_lists_four_models_with_scores():
    assert [m["key"] for m in R["models"]] == [k for k, _, _ in MODEL_KEYS]
    for m in R["models"]:
        for d in P["depths"]:
            assert "holdout" in R["metrics"][m["key"]][d]


def test_presets_cover_every_input():
    for p in R["presets"].values():
        assert set(p["values"]) == set(R["inputs"])


def test_presets_stay_inside_training_range():
    for p in R["presets"].values():
        for f, v in p["values"].items():
            lo, hi = R["input_meta"][f]["range"]
            assert lo - 1e-6 <= v <= hi + 1e-6, f
