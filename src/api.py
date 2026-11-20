"""Web app + REST API for the six-depth soil-moisture chain.

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


def _load(kind, d):
    m = xgb.XGBRegressor()
    m.load_model(MODELS_DIR / kind / f"{d}.json")
    return m


MODELS = {k: {d: _load(k, d) for d in DEPTHS} for k in ("independent", "chained")}

app = FastAPI(
    title="Multi-layer Root-Zone Soil Moisture API",
    description="Predicts volumetric soil moisture (cm³/cm³) at 30, 60, 90, 120, 150 and 180 cm from "
                "PlanetScope, weather, soil and terrain inputs. Each layer's prediction is fed to the "
                "model of the layer below (Lamichhane et al., 2025, Vadose Zone Journal).",
    version="1.0.0",
)


class ProfileRequest(BaseModel):
    inputs: dict[str, float] = Field(..., description="Input name → value. GET /meta lists all 30 inputs.")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "depths": DEPTHS, "chain_mode": REG["chain_mode"]}


@app.get("/meta")
def meta():
    """Inputs, valid ranges, presets, validation scores and drivers for every depth."""
    return REG


@app.post("/predict")
def predict(req: ProfileRequest):
    """Run the full chain top-down and return the whole soil-moisture profile."""
    missing = [f for f in REG["inputs"] if f not in req.inputs]
    if missing:
        raise HTTPException(422, f"Missing inputs: {', '.join(missing)}")
    warnings = []
    for f in REG["inputs"]:
        v, m = req.inputs[f], REG["input_meta"][f]
        if not math.isfinite(v):
            raise HTTPException(422, f"{f} must be a finite number")
        lo, hi = m["limits"]
        if not lo <= v <= hi:
            raise HTTPException(422, f"{f} = {v} is not physically possible (allowed {lo} to {hi})")
        rlo, rhi = m["range"]
        if not rlo <= v <= rhi:
            warnings.append(f"{m['label']} = {v} is outside the training range ({rlo} to {rhi}).")

    X = pd.DataFrame([{f: req.inputs[f] for f in REG["inputs"]}])
    layers, feed = [], None
    for i, d in enumerate(DEPTHS):
        Xd = X[FEATS[d]]
        indep = float(MODELS["independent"][d].predict(Xd)[0])
        upper_contribution = 0.0
        if i == 0:
            chained = indep
        else:
            Xc = Xd.assign(**{UPPER: feed})
            chained = float(MODELS["chained"][d].predict(Xc)[0])
            contribs = MODELS["chained"][d].get_booster().predict(xgb.DMatrix(Xc), pred_contribs=True)[0]
            upper_contribution = float(contribs[list(Xc.columns).index(UPPER)])
        layers.append({
            "depth": d, "depth_cm": int(d[1:]),
            "chained": round(chained, 4), "independent": round(indep, 4),
            "fed_from_above": None if feed is None else round(float(feed), 4),
            "upper_layer_effect": round(upper_contribution, 4),
            "holdout_r2": REG["metrics"][d]["chained"]["holdout"]["r2"],
        })
        feed = chained
    return {"unit": "cm³/cm³", "chain_mode": REG["chain_mode"], "layers": layers, "warnings": warnings}
