import streamlit as st
import pandas as pd, os
from src.model_utils import load_prophet_model, load_lstm_model, forecast_prophet, forecast_lstm

st.set_page_config(page_title='Stock Forecast', layout='wide')
st.title('📈 Stock Forecasting (Prophet + LSTM)')

ticker = st.text_input('Ticker', 'AAPL')
days = st.slider('Forecast days', 7, 90, 30)
model_choice = st.selectbox('Model', ['Prophet','LSTM'])

if st.button('Run Forecast'):
    data_path = os.path.join('data', f'{ticker}.csv')
    if not os.path.exists(data_path):
        st.info('Fetching data...')
        from src.fetch_data import fetch_stock_data
        fetch_stock_data(ticker=ticker, save_path=data_path)
    df = pd.read_csv(data_path)
    st.subheader('Recent close prices')
    st.line_chart(df.set_index('Date')['Close'].tail(200))
    if model_choice == 'Prophet':
        m = load_prophet_model()
        res = forecast_prophet(m, days)
        st.subheader('Prophet forecast')
        st.line_chart(res.set_index('ds')['yhat'])
        st.table(res)
    else:
        model, scaler = load_lstm_model()
        preds = forecast_lstm(model, scaler, df, days)
        st.subheader('LSTM forecast')
        st.line_chart(preds)
        st.table(pd.DataFrame({'pred':preds}))
