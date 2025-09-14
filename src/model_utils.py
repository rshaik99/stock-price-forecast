import joblib, pandas as pd, os, numpy as np
from tensorflow.keras.models import load_model

def load_prophet_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'prophet_model.pkl')
    return joblib.load(path)

def load_lstm_model(model_path=None, scaler_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm_model.h5')
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm_scaler.pkl')
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def forecast_prophet(model, days=30):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days)

def forecast_lstm(model, scaler, df, days=30):
    prices = df['Close'].values.reshape(-1,1)
    last_60 = scaler.transform(prices)[-60:]
    preds = []
    for _ in range(days):
        X = last_60.reshape(1,60,1)
        p = model.predict(X, verbose=0)
        preds.append(p[0][0])
        last_60 = np.vstack([last_60[1:], p])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds.flatten()
