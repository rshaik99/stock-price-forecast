from fastapi import FastAPI, HTTPException
import pandas as pd, os
from model_utils import load_prophet_model, load_lstm_model, forecast_prophet, forecast_lstm
from logger import get_logger

logger = get_logger(__name__)
app = FastAPI()

@app.get('/')
def root():
    return {'status':'ok', 'message':'Stock Forecasting API'}

@app.get('/forecast/prophet')
def forecast_prophet_api(days: int = 30, ticker: str = 'AAPL'):
    try:
        model = load_prophet_model()
        res = forecast_prophet(model, days)
        return res.to_dict(orient='records')
    except Exception as e:
        logger.exception('Prophet forecast failed')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/forecast/lstm')
def forecast_lstm_api(days: int = 30, data_path: str = None):
    try:
        model, scaler = load_lstm_model()
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'AAPL.csv')
        df = pd.read_csv(data_path)
        preds = forecast_lstm(model, scaler, df, days)
        return {'predictions': preds.tolist()}
    except Exception as e:
        logger.exception('LSTM forecast failed')
        raise HTTPException(status_code=500, detail=str(e))
