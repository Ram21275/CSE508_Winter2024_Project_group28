import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

plt.style.use('dark_background')

model = load_model('/Users/ramdabas/Downloads/Stock_Market_Prediction_ML 2/Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Avg 50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,6))
plt.plot(ma_50_days, 'r--', label='MA 50 days')
plt.plot(data.Close, 'g-', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Averages of Stock Prices')
plt.legend(loc='upper left')
st.pyplot(fig1)

st.subheader('Price vs Moving Avg 50 vs Moving Avg 100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,6))
plt.plot(ma_50_days, 'r--', label='MA 50 days')
plt.plot(ma_100_days, 'b:', label='MA 100 days')
plt.plot(data.Close, 'g-', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Averages of Stock Prices')
plt.legend(loc='upper left')
st.pyplot(fig2)

st.subheader('Price vs Moving Avg 100 vs Moving Avg 200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,6))
plt.plot(ma_100_days, 'r--', label='MA 100 days')
plt.plot(ma_200_days, 'b:', label='MA 200 days')
plt.plot(data.Close, 'g-', label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Averages of Stock Prices')
plt.legend(loc='upper left')
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'b', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original Price vs Predicted Price')
plt.legend(loc='upper left')
st.pyplot(fig4)
