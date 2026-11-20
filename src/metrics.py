"""R², RMSE and RRMSE (%), as in the paper."""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(y_true)
    y_true, y_pred = y_true[ok], y_pred[ok]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": round(float(r2_score(y_true, y_pred)), 3), "rmse": round(rmse, 4),
            "rrmse_pct": round(rmse / float(np.mean(y_true)) * 100, 2), "n": int(ok.sum())}
