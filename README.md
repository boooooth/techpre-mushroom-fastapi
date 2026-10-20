# Mushroom Toxicity Classification API

A clean, modular Machine Learning project built with **scikit-learn** and served with **FastAPI**.

Given the physical attributes of a wild mushroom (odor, cap shape, gill attachment, spore print, etc.), the system predicts whether it is **edible** or **poisonous** with a safety-first decision threshold.

```
Mushroom/
├── data/
│   └── mushrooms.csv             # 8,124 mushroom records from the Audubon Society Field Guide
├── models/
│   └── mushroom.joblib           # Fitted model pipeline + safety threshold + metadata bundle
├── app/
│   ├── schemas.py                # Pydantic models for request validation & OpenAPI documentation
│   ├── ml.py                     # Framework-agnostic prediction & risk evaluation routines
│   └── main.py                   # FastAPI application with startup lifespan & endpoints
├── train_mushroom.py             # Full training pipeline (EDA, baselines, CV, threshold tuning)
└── requests.http                 # Ready-made requests for VS Code REST Client / JetBrains
```

## Setup & Running

Activate your virtual environment with scikit-learn and FastAPI installed:

```bash
# Train the model (takes ~3 seconds)
python train_mushroom.py

# Start the API server with auto-reload
fastapi dev app/main.py --port 8001
```

Once running, open your browser to **http://127.0.0.1:8001/docs** for the interactive Swagger documentation.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service heartbeat & verification of loaded models |
| `GET` | `/models` | Audit trail: algorithm, training timestamp, and accuracy/recall metrics |
| `POST` | `/mushroom/predict` | Single mushroom classification: returns `is_poisonous`, `risk`, and recommendations |
| `POST` | `/mushroom/predict/batch` | High-performance vectorized classification for multiple specimens in one call |

## Safety-First ML Architecture

* **Zero False Negatives**: In mushroom foraging, misclassifying a poisonous mushroom as edible is lethal. The decision threshold (0.68) was selected on cross-validated training folds to guarantee **0 false negatives**.
* **One-Hot Encoding with Unknown Handling**: Categorical strings are transformed using `OneHotEncoder(handle_unknown='ignore')`, preventing server crashes on unexpected input.
