import yfinance as yf
import pandas as pd
import argparse, os
from logger import get_logger

logger = get_logger(__name__)

def fetch_stock_data(ticker='AAPL', period='5y', interval='1d', save_path=None):
    df = yf.download(ticker, period=period, interval=interval)
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{ticker}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    logger.info(f"Saved data for {ticker} at {save_path}")
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--period', default='5y')
    p.add_argument('--interval', default='1d')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    fetch_stock_data(args.ticker, args.period, args.interval, args.out)
