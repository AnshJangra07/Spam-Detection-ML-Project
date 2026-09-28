# Spam-Ham Detection

A text-classification project that labels messages as spam or ham. Training data is read from MongoDB, validated and transformed locally, used to select a classifier, evaluated against the current S3 model, and published to S3 when accepted. A FastAPI application serves the prediction form and uses the published model for inference.

## Architecture

![Training pipeline](flowchart/training%20pipeline.png)

![Prediction pipeline](flowchart/prediction%20pipeline.png)

The pipeline is divided into data ingestion, schema validation, text transformation, model selection, evaluation, and model pushing. The root `spamham.csv` currently has 10,162 rows with source columns `Label` and `Message`; ingestion maps these to `class` and `message`.

## Technology

- Python, pandas, scikit-learn, NLTK, and `neuro-mf`
- MongoDB for training-data ingestion
- AWS S3 for model storage
- FastAPI, Uvicorn, Jinja2, and static HTML/CSS for the web app

## Setup

Use Python 3.11 and create/activate a virtual environment:

```powershell
py -3.11 -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root. The current code reads these exact variable names:

```dotenv
MONGODB_URL_KEY=mongodb+srv://<user>:<password>@<cluster>/
AWS_ACCESS_KEY_ID_ENV_KEY=<aws-access-key-id>
AWS_SECRET_ACCESS_KEY_ENV_KEY=<aws-secret-access-key>
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

The MongoDB database and collection are `spam_ham_database` and `spam_ham_collection`. AWS credentials need permission to read and write the configured S3 bucket, `spam-detection-model2026`, in `us-east-1`. Keep `.env` out of version control and never commit credentials.

## Training

Load the dataset into MongoDB once, then run the training pipeline:

```powershell
python upload_data_mongodb.py
python demo.py
```

`upload_data_mongodb.py` inserts documents; avoid rerunning it against an already populated collection unless duplicates are intended. The pipeline writes timestamped artifacts beneath `src/artifact/`. Model selection uses cross-validated spam-class F1, and the trainer logs held-out accuracy, precision, recall, and F1. A candidate is pushed only if it meets the configured minimum F1 and is new or improves on the current S3 model by at least the configured threshold.

## Web App

Start the API from the project root:

```powershell
python app.py
```

Open `http://127.0.0.1:5001`. Available routes:

| Method | Path            | Purpose                                   |
| ------ | --------------- | ----------------------------------------- |
| `GET`  | `/`             | Home page                                 |
| `GET`  | `/predict`      | Prediction form                           |
| `POST` | `/predict`      | Predict from the `input_text` form field  |
| `POST` | `/train`        | Start the training pipeline in background |
| `GET`  | `/train/status` | Read training state and held-out metrics  |

The home page shows training progress and, when complete, held-out accuracy, precision, recall, and F1. Training and prediction require their configured MongoDB/AWS services and data/model artifacts. Prediction uses the latest model published to S3.

## Configuration

- `config/schema.yaml`: required training feature and target columns
- `config/prediction_schema.yml`: prediction input/output column metadata
- `config/model.yaml`: estimator candidates, parameter grids, cross-validation, and F1 scoring
- `src/constant/`: pipeline settings, column names, bucket, and service configuration

## Project Structure

```text
config/                 Schema and model-selection settings
src/components/         Ingestion, validation, transformation, training, evaluation, pushing
src/data_access/        MongoDB dataset access
src/ml/model/           Model wrapper, S3 loading, shared text preprocessing
src/pipeline/           Training and prediction orchestration
templates/              FastAPI/Jinja pages
static/                 CSS and image assets
notebooks/              Exploration and model-training notebooks
```

## Metrics

Metrics are calculated from each training run's held-out test split and included in the returned model-trainer artifact and training log. No fixed benchmark scores are listed here because they should be copied from a verified run, not estimated.
