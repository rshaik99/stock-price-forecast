import pandas as pd, joblib, os, argparse
from prophet import Prophet
from logger import get_logger

logger = get_logger(__name__)

def train_prophet(data_path=None, save_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'AAPL.csv')
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'prophet_model.pkl')
    df = pd.read_csv(data_path)
    df = df.rename(columns={'Date':'ds', 'Close':'y'})[['ds','y']].dropna()
    m = Prophet()
    m.fit(df)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(m, save_path)
    logger.info(f"Prophet model saved to {save_path}")
    return m

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    train_prophet(args.data, args.out)
