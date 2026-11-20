"""Web app + REST API for the four six-depth soil-moisture models.

Run locally (from the project folder):
    uvicorn api:app --reload --app-dir src
Then open http://127.0.0.1:8000 (web app) or http://127.0.0.1:8000/docs (API).

Only models/ (registry.json + XGBoost .json files) is needed: no data, no training code.
"""
import json
import math
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

REG = json.loads((MODELS_DIR / "registry.json").read_text())
DEPTHS, FEATS, UPPER = REG["depths"], REG["features"], REG["upper_col"]
MODEL_LIST = {m["key"]: m for m in REG["models"]}


def _load(variant, kind, d):
    m = xgb.XGBRegressor()
    m.load_model(MODELS_DIR / variant / kind / f"{d}.json")
    return m


# MODELS[variant][kind][depth]
MODELS = {v: {k: {d: _load(v, k, d) for d in DEPTHS} for k in ("independent", "chained")} for v in REG["variants"]}

app = FastAPI(
    title="Multilayer Soil Moisture API",
    description="Predicts volumetric soil moisture (cm³/cm³) at 30, 60, 90, 120, 150 and 180 cm from "
                "PlanetScope, climate, soil and elevation inputs. Four models: with or without soil properties, "
                "and with or without the predicted moisture of the layer above "
                "(Lamichhane et al., 2025, Vadose Zone Journal).",
    version="2.0.0",
)


class ProfileRequest(BaseModel):
    inputs: dict[str, float] = Field(..., description="Input name → value. GET /meta lists the inputs each model needs.")
    model: str = Field("with_soil_chained", description="with_soil_chained (paper), with_soil_independent, "
                                                        "no_soil_chained or no_soil_independent")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "depths": DEPTHS, "models": list(MODEL_LIST), "chain_mode": REG["chain_mode"]}


@app.get("/meta")
def meta():
    """Models, inputs, valid ranges, presets, validation scores and drivers for every depth."""
    return REG


def needed_inputs(variant):
    return sorted({f for d in DEPTHS for f in FEATS[variant][d]}, key=REG["inputs"].index)


def run_profile(X, variant, kind):
    """Run one model top-down; returns one dict per depth."""
    layers, feed = [], None
    for i, d in enumerate(DEPTHS):
        Xd = X[FEATS[variant][d]]
        effect = None
        if kind == "chained" and i > 0:
            Xd = Xd.assign(**{UPPER: feed})
            contribs = MODELS[variant][kind][d].get_booster().predict(xgb.DMatrix(Xd), pred_contribs=True)[0]
            effect = round(float(contribs[list(Xd.columns).index(UPPER)]), 4)
        value = float(MODELS[variant][kind][d].predict(Xd)[0])
        layers.append({
            "depth": d, "depth_cm": int(d[1:]), "value": round(value, 4),
            "fed_from_above": None if (kind != "chained" or feed is None) else round(float(feed), 4),
            "upper_layer_effect": effect,
            "holdout_r2": REG["metrics"][f"{variant}_{kind}"][d]["holdout"]["r2"],
        })
        feed = value
    return layers


@app.post("/predict")
def predict(req: ProfileRequest):
    """Run the chosen model top-down. Also returns the profile of every other model whose inputs were given."""
    if req.model not in MODEL_LIST:
        raise HTTPException(422, f"Unknown model '{req.model}'. Choose one of: {', '.join(MODEL_LIST)}")
    chosen = MODEL_LIST[req.model]
    missing = [f for f in needed_inputs(chosen["variant"]) if f not in req.inputs]
    if missing:
        raise HTTPException(422, f"Missing inputs for {req.model}: {', '.join(missing)}")
    warnings = []
    for f, v in req.inputs.items():
        if f not in REG["input_meta"]:
            continue
        m = REG["input_meta"][f]
        if not math.isfinite(v):
            raise HTTPException(422, f"{f} must be a finite number")
        lo, hi = m["limits"]
        if not lo <= v <= hi:
            raise HTTPException(422, f"{f} = {v} is not physically possible (allowed {lo} to {hi})")
        rlo, rhi = m["range"]
        if not rlo <= v <= rhi and f in needed_inputs(chosen["variant"]):
            warnings.append(f"{m['label']} = {v} is outside the training range ({rlo} to {rhi}).")

    X = pd.DataFrame([{f: v for f, v in req.inputs.items() if f in REG["input_meta"]}])
    layers = run_profile(X, chosen["variant"], chosen["kind"])
    compare = {}
    for key, m in MODEL_LIST.items():
        if all(f in req.inputs for f in needed_inputs(m["variant"])):
            compare[key] = layers if key == req.model else run_profile(X, m["variant"], m["kind"])
    return {"unit": "cm³/cm³", "model": req.model, "label": chosen["label"], "layers": layers,
            "compare": {k: [L["value"] for L in v] for k, v in compare.items()}, "warnings": warnings}
