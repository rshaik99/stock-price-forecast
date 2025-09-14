import pandas as pd, os, joblib, argparse, numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from logger import get_logger

logger = get_logger(__name__)

def train_lstm(data_path=None, save_path=None, scaler_path=None, epochs=5):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'AAPL.csv')
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm_model.h5')
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm_scaler.pkl')

    df = pd.read_csv(data_path)
    prices = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"LSTM model saved to {save_path} and scaler to {scaler_path}")
    return model, scaler

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--scaler', default=None)
    p.add_argument('--epochs', default=5, type=int)
    args = p.parse_args()
    train_lstm(args.data, args.out, args.scaler, args.epochs)
