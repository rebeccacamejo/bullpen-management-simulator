# ⚾ Bullpen Management Simulator (BMS)

The Bullpen Management Simulator (BMS) is an end‑to‑end machine‑learning application that recommends the most appropriate reliever to bring into a game given the **current game state** and each candidate’s rest/fatigue, handedness and platoon match‑up.  It combines modern sabermetric principles with a simple user interface and full MLOps packaging so that you can train, serve, monitor and improve the model just like a production service.

## 💡 Motivation

Traditional baseball statistics (ERA, saves, holds) don’t reflect the contextual complexity facing a manager when choosing a reliever.  **Tom Tango’s book “The Book”** showed that run expectancy and leverage matter more than raw stats, and that situational decisions cannot be evaluated in isolation.  Modern run value metrics, such as those on Baseball Savant and FanGraphs, calculate cumulative pitch values from run expectancy matrices, but they are geared toward batters and aren’t integrated into an interactive decision tool for bullpen management.  Existing WAR models (FIP‑based WAR, rWAR, DRA‑based WARP) estimate a pitcher’s overall season value but do not tell you who to bring into a high‑leverage situation.  Technologies such as TrackMan and Rapsodo provide rich pitch‑tracking data, but coaches still need a structured way to apply that data to game decisions.

This project was born from a frustration with the wealth of stats available to fans and analysts and the lack of a simple tool that answers the question: **“Who should pitch next?”**  As a baseball fan obsessed with how pitchers control the game, I wanted to build a system that translates pitch‑level data and context into actionable insights for managers, coaches, analysts and fans.

## 🚀 What it does

- **Predicts expected runs** allowed over the next `K` batters (default three) using a supervised regression model trained on historical play‑by‑play data.
- **Recommends the reliever** with the lowest adjusted expected runs, applying penalties for back‑to‑back days, high pitch counts in the previous outing or unavailability.
- **Explains its reasoning** by returning both the raw predicted value and the penalty components for each candidate.
- **Offers an interactive web UI** built with Streamlit so that users can tweak game state parameters and bullpen options and see the recommendation instantly.
- **Implements MLOps best practices**: MLflow for experiment tracking and model registry, Prometheus for service metrics, Evidently for drift detection, Great Expectations for data quality checks, and GitHub Actions for CI.
- **Ships as a containerised microservice** with a FastAPI back‑end and a separate UI container, ready for deployment on platforms such as Google Cloud Run or AWS ECS.

## 🧠 How it works

### Target and Model

The core problem is framed as a **regression**: given a game state and reliever characteristics, estimate the **expected number of runs** the opposing team will score over the next `K` batters (or until the inning ends).  This is a more stable and interpretable target than win probability because runs translate directly into win outcomes and can be summed over multiple situations.  The model uses a **Gradient Boosting Regressor** (via scikit‑learn) as a strong baseline for tabular data with limited features and moderate non‑linearities.  Gradient‑boosting models handle interactions (e.g. the effect of runners on base and inning) well, are easy to fit on relatively small datasets, and can be interpreted with tools like SHAP.

Features include:

* **Game context**: number of outs, base/out state (encoded as runners on base), inning, score differential, a simple leverage proxy (late inning indicator × close game indicator), home/away and park identifier.
* **Match‑up/platoon**: whether the reliever’s handedness matches the expected batter handedness (estimated with a simple segment indicator: top, middle or bottom of the lineup).
* **Reliever fatigue**: days since last outing and pitch count in the previous outing.
* **Optional override**: future extensions might incorporate batter quality, precise leverage index or pitch‑tracking metrics (velocity, movement), but these are outside the current scope for simplicity.

During recommendation, the model prediction is adjusted by **usage penalties**: relievers get a positive penalty if they pitched the day before or threw more than 20 pitches in their last outing, and an infinite penalty if they are marked unavailable.  The reliever with the lowest penalized expected runs is recommended.

### Data and Labeling

The model is trained on pitch‑level data pulled from Statcast via the **pybaseball** library.  For each reliever appearance, we derive features from the game state just before they enter and compute the **actual runs allowed** over the next `K` batters.  This is similar to run expectancy values used in Tom Tango’s book but tailored for bullpen management.  A separate script prepares the dataset, computes features and labels, and logs the training run to MLflow.

### MLOps and Path to Production

- **Experiment tracking and model registry** with MLflow: every training run records parameters (e.g. number of estimators, depth) and performance metrics (e.g. validation MAE) and saves the trained model artifact.  The best model is registered and loaded by the API.
- **Service metrics** using Prometheus: the API exposes `/metrics` that reports request counts, latency and online mean absolute error (mae).  These metrics can be scraped by Prometheus and visualised in Grafana.
- **Data and model drift** detection with Evidently: a nightly Prefect job loads recent predictions and outcomes, compares feature distributions and performance against a reference window, and writes an HTML report while updating Prometheus gauges.
- **Data quality** checks with Great Expectations: ingestion scripts validate that numeric features are within expected ranges and that categorical fields contain valid values.  If validation fails, the batch is rejected.
- **Continuous integration**: GitHub Actions runs code formatting, linting, static type checking and smoke tests on every push.
- **Containerization**: separate Dockerfiles for the API and UI.  A `docker-compose.yml` orchestrates both services for local development.  Cloud‑native deployment instructions are included in the README.

## 📦 Repository structure

```
bullpen-management-simulator/
├─ README.md
├─ requirements.txt
├─ Makefile
├─ .github/workflows/ci.yml
├─ bms/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ domain.py
│  ├─ features.py
│  ├─ model_expected_runs.py
│  ├─ recommender.py
├─ api/
│  ├─ main.py
│  ├─ schemas.py
│  ├─ metrics.py
│  └─ Dockerfile
├─ ui_streamlit/
│  ├─ app.py
│  └─ Dockerfile
├─ ops/
│  ├─ drift_job.py
│  └─ gx_checks.py
├─ docker-compose.yml
└─ models/
    └─ (trained model artifacts)
```

## 👥 User stories

* **Manager/Bench Coach**: “Given men on first and third with one out in the eighth and our top relievers spent, who minimizes expected runs over the next three batters?”
* **Pitching Coach**: “If we avoid using our closer today after he threw 25 pitches last night, what is the cost in expected runs and who should we use instead?”
* **Front‑office Analyst**: “How do penalty adjustments (rest/fatigue) versus pure match‑up change the recommendation?  Which of our relievers is most efficient in high leverage?”
* **Fan/Broadcaster**: “I want to interactively explore how different game states and bullpen choices affect the outcome.”

## 🧪 Running the project

### Local development

1. **Install dependencies**:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train a demo model** (using synthetic data or a small Statcast sample):

   ```bash
   python scripts/train_demo.py
   ```

   This generates `models/bms.joblib` and logs the run to MLflow.

3. **Run API and UI locally** in separate terminals:

   ```bash
   API_PORT=8080 uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
   ```

   ```bash
   API_URL=http://localhost:8080 streamlit run ui_streamlit/app.py
   ```

   Navigate to `http://localhost:8501` to interact with the simulator.

### Docker compose

To launch both services together:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8080` and the UI at `http://localhost:8501`.

### Cloud deployment

1. Build and push the API and UI images:

   ```bash
   gcloud builds submit api --tag gcr.io/PROJECT_ID/bms-api
   gcloud builds submit ui_streamlit --tag gcr.io/PROJECT_ID/bms-ui
   ```

2. Deploy to Cloud Run:

   ```bash
   gcloud run deploy bms-api --image gcr.io/PROJECT_ID/bms-api --region=us-central1 --allow-unauthenticated
   gcloud run deploy bms-ui --image gcr.io/PROJECT_ID/bms-ui --region=us-central1 --allow-unauthenticated \
     --set-env-vars API_URL=https://bms-api-<hash>-uc.a.run.app
   ```

## 📖 References

* Tom Tango et al., *The Book: Playing the Percentages in Baseball*.  Chapters on run expectancy and game state emphasise the importance of leverage and context.
* Baseball Savant’s run value metric description: run values are cumulative sums of pitch outcomes relative to run expectancy.
* FanGraphs on different pitching WAR models: comparisons of fWAR (FIP‑based), rWAR (runs allowed) and DRA‑based WARP.
* Pitch tracking technology overview (Rapsodo, TrackMan, etc.): describes the data available to coaches and scouts.
