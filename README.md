# deposits

# Deposits Forecast Model

This project is a small end‑to‑end application for forecasting customer deposits based on historical data.  
It combines a scikit‑learn forecasting pipeline with a simple web interface and can be run locally or in Docker.

---

## Features

- Upload historical deposits data from Excel.
- Preprocess and transform time series features.
- Forecast future deposits using a trained k‑nearest neighbors (kNN) model.
- Download the forecast results back to Excel.
- Run as a local Flask web app or in a Docker container.

---

## Project structure

- `depositsweb.py` – main Flask application with web UI and routing.
- `universal_forecast.py` – forecasting logic and model interface.
- `knn_full_components.pkl` – serialized scikit‑learn pipeline and kNN model.
- `templates/` – HTML templates for the web interface.
- `static/` – CSS and static assets.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – container definition.
- `.dockerignore` – files excluded from the build context.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/corifinusica/deposits.git
cd deposits
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
python depositsweb.py
```

By default the app will start on `http://127.0.0.1:5000/` (or the host/port defined inside `depositsweb.py`).  
Open this URL in your browser to access the web interface.

---

## Usage

1. Prepare your historical deposits data in Excel format (see the example file structure used in the app).
2. Open the web UI.
3. Upload your Excel file with historical deposits.
4. The app will preprocess the data and generate a forecast using the trained kNN model.
5. Review the forecast on the page and/or download the results as an Excel file.

The exact expected column names and date format follow the structure used during model training.  
Adjust your input data accordingly if needed.

---

## Running with Docker

You can also build and run the application in a Docker container.

### Build the image

```bash
docker build -t deposits-forecast .
```

### Run the container

```bash
docker run -p 5000:5000 deposits-forecast
```

Then open `http://127.0.0.1:5000/` in your browser.

---

## Model details

- Framework: scikit‑learn
- Algorithm: k‑nearest neighbors (kNN) regressor as part of a preprocessing pipeline
- Serialization: `joblib` / `pickle` (`knn_full_components.pkl`)

The model was trained and evaluated offline and then exported as a single pipeline object.  
The web app loads this pipeline at startup and applies it to new input data.

---

## Tech stack

- Python
- Flask
- scikit‑learn
- pandas
- openpyxl (Excel support)
- HTML / CSS (Flask templates)
- Docker

---

## Future improvements

Potential next steps for this project:

- Add input validation and clearer error messages in the UI.
- Expose basic evaluation metrics and plots for the training data.
- Support multiple forecasting horizons and model comparison.
- Add configuration for different input file formats.
```

