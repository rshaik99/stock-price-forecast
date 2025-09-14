# Minimal EDA script (also available as notebook)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Close'].plot(title='Close Price History', figsize=(10,4))
df['Close'].rolling(20).mean().plot(label='MA20')
plt.legend()
plt.show()
