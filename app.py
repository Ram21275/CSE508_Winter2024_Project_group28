import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

gpu_no = 0
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# device

sns.set(style='darkgrid')

model_path_sent = './saved_model/'
tokenizer = AutoTokenizer.from_pretrained(model_path_sent)
model_sent = AutoModelForSequenceClassification.from_pretrained(model_path_sent)
model_sent.to(device)

import praw
reddit = praw.Reddit(
    client_id='MazNPVF4wTFD378C0wVJHg',
    client_secret='02sXZyN4qv1z6n5wrlIP0gBDCVtGtQ',
    user_agent='Sentiment analysis',
)



def get_model_op(text, tokenizer = tokenizer, model = model_sent):
    label_text = ['postive', 'negative','neutral']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    # print(inputs)
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)
    # print(probabilities)
    predicted_class = logits.argmax().item()
    # predicted_label = params["index_to_label"].get(predicted_class, 'Unknown')
    return label_text[predicted_class]#, round(probabilities[0][predicted_class].item(), 2),predicted_class,

def display_reddit_posts_and_comments():
    global stock
    # stock = st.text_input('Enter Stock Symbol', 'AAPL').upper()

    if stock:
        st.header(f'Top Reddit posts and sentiment analysis for {stock}')
        try:
            subreddit = reddit.subreddit("all") 
            posts = subreddit.search(stock, limit=5)

            for post in posts:
                with st.expander(f"{post.title}"):
                    st.markdown(f"**Link:** [Here]({post.url})")
                    st.markdown(f"**Post Content:** {post.selftext[:1000]}...")  
                    sentiment = get_model_op(post.title + ' ' + post.selftext)
                    st.markdown(f"**Post Sentiment:** {sentiment}")
                    post.comments.replace_more(limit=0)
                    comments = post.comments.list()[:5]  # Top 5 comments
                    for comment in comments:
                        st.markdown(f"**Comment:** {comment.body[:1000]}...") 
                        comment_sentiment = get_model_op(comment.body)
                        st.markdown(f"**Comment Sentiment:** {comment_sentiment}")
        except Exception as e:
            st.error("Failed to fetch Reddit data. Check your credentials and network connection.")


plt.style.use('dark_background')

model = load_model('./Stock Predictions Model.keras')
st.title('Multi-Modal Stock Information Retrival and Prediction')

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2024-04-22'
display_reddit_posts_and_comments()
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
