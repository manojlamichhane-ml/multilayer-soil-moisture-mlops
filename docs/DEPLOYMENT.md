# Deployment guide

Everything below is run from `D:\MLOPS\Multilayer_Soil_Moisture` in a VS Code terminal (Command Prompt). It takes about 30 minutes the first time.

```
Part 1  Laptop:  new Python 3.12 environment, check the pipeline and the app
Part 2  GitHub:  new empty repository, push the project
Part 3  Render:  new web service from that repository → public link
```

This project is completely separate from `soil-moisture-mlops` and from the original `multilayer-soil-moisture-ml` repository. Neither is touched.

---

## Part 1: laptop

### 1.1 Open the project
In VS Code: **File → Open Folder** → `D:\MLOPS\Multilayer_Soil_Moisture`. Open a terminal with **Ctrl+`**.

### 1.2 Create a new environment (Python 3.12)
XGBoost 3.4 needs Python 3.12, so this project gets its own environment instead of reusing `surface_SM` (3.10). One environment per project is good practice anyway.

```
conda create -n multilayer_SM python=3.12 -y
conda activate multilayer_SM
pip install -r requirements.txt
```

The prompt should now start with `(multilayer_SM)`. In VS Code, also press **Ctrl+Shift+P → Python: Select Interpreter → multilayer_SM**.

### 1.3 Check the pipeline
```
git init
dvc status
pytest -v
```
- `git init` turns the folder into a Git project (DVC needs it). DVC itself is already set up (the `.dvc` folder).
- `dvc status` should say `Data and pipelines are up to date.`: the models were trained from your data and their fingerprints are in `dvc.lock`.
- `pytest -v` should end with `33 passed`.

To retrain everything yourself (about 1.5 minutes): `dvc repro --force`. This also fills your local MLflow logbook; view it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### 1.4 Try the app
```
uvicorn api:app --reload --app-dir src
```
Open http://127.0.0.1:8000. Click the presets and **Drop it through the profile**. Stop with **Ctrl+C**.

---

## Part 2: GitHub

### 2.1 Create an empty repository
On github.com (signed in as **manojlamichhane-ml**): **+ → New repository**
- Name: `multilayer-soil-moisture-mlops`
- Public
- Do **not** add a README, .gitignore or license

### 2.2 Check what will be uploaded
```
git status --short
```
The list must **not** contain `data/` or `mlflow.db`. Both are in `.gitignore`, so your private data stays on your laptop. It should contain `src/`, `models/`, `reports/`, `tests/`, `params.yaml`, `dvc.yaml`, `dvc.lock`, `Dockerfile`, `README.md`, `docs/`, `.github/`.

> **`.github` folder:** if `.github\workflows\ci.yml` is missing, it was placed in the main folder as `ci.yml` (remote tools cannot write into `.github`). Move it:
> ```
> mkdir .github\workflows
> move ci.yml .github\workflows\ci.yml
> ```

### 2.3 First snapshot and push
```
git add .
git commit -m "Multi-layer soil moisture MLOps: chained XGBoost, DVC, MLflow, FastAPI, Docker"
git branch -M main
git remote add origin https://github.com/manojlamichhane-ml/multilayer-soil-moisture-mlops.git
git push -u origin main
```
If a sign-in window appears, sign in as **manojlamichhane-ml** (the account that owns this repository).

Open the repository page: the README shows the animated cascade. The **Actions** tab shows the tests running; a green tick means they passed.

---

## Part 3: Render

### 3.1 New web service
On render.com → **+ New → Web Service** → choose **multilayer-soil-moisture-mlops**. If it isn't listed, click **Configure GitHub** and give Render access to this repository.

### 3.2 Settings
| Setting | Value |
|---|---|
| Name | `multilayer-soil-moisture` (this becomes the web address) |
| Language / Runtime | **Docker** (detected from the Dockerfile) |
| Branch | `main` |
| Region | Oregon (US West) |
| Root Directory | leave empty |
| Instance type | **Free** ($0 / month). Render may pre-select a paid plan; click the Free row |

Click **Deploy Web Service**.

### 3.3 Wait for “Your service is live”
The log shows Docker installing Python 3.12 and the packages (2–4 minutes). Then open:

- the app: `https://multilayer-soil-moisture-mlops.onrender.com`
- the API docs: `https://multilayer-soil-moisture-mlops.onrender.com/docs`

If Render gave a different address (for example with a suffix), update the two links at the top of `README.md`, then commit and push.

### 3.4 How updates work from now on
Every `git push` to `main` makes GitHub run the tests and Render rebuild the app automatically. A typical change:
```
(edit params.yaml)
dvc repro
pytest -v
git add .
git commit -m "describe the change"
git push
```

### Good to know
- The free instance sleeps after 15 minutes without visitors; the next visit takes about a minute to wake it.
- If a deploy fails, open **Logs** on Render. The most common causes are a typo in `requirements-api.txt` or a missing file in `models/`.
- No data is ever uploaded: the container holds only `src/api.py`, the web page and the trained models.
