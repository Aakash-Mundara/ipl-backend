# Cricket Player Performance Prediction API

## Overview
A FastAPI backend that predicts cricket batter run scores using a pre-trained machine learning model. It analyzes historical match data and current form to generate predictions with explanations.

## Tech Stack
- **Language**: Python 3.12
- **Web Framework**: FastAPI with Uvicorn ASGI server
- **ML**: scikit-learn (model loaded from `model.pkl`)
- **Data**: Pandas/NumPy processing `deliveries.csv` and `matches.csv`

## Project Structure
- `main.py` - FastAPI app with `/predict` endpoint and feature engineering logic
- `model.pkl` - Pre-trained scikit-learn model
- `deliveries.csv` - Ball-by-ball delivery data
- `matches.csv` - Match metadata
- `requirements.txt` - Python dependencies

## Running the App
The API runs on port 8000:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints
- `GET /` - Health check
- `POST /predict` - Predict runs for a player given team, opponent, venue, mean_runs, boundary_pct, strike_rate

## Workflow
- **Start application**: Runs the uvicorn server on port 8000 (console output)
