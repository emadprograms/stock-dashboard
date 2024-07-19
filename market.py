import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from dotenv import load_dotenv
import os


# Function to fetch and prepare data for a list of tickers from yfinance
def fetch_data(tickers):
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, period="5y")
        stock_data['Ticker'] = ticker
        stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()
        stock_data['SMA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
        stock_data['RSI'] = compute_rsi(stock_data['Adj Close'])
        stock_data['Volatility'] = stock_data['Adj Close'].pct_change().rolling(window=21).std()
        stock_data['High_Volatility'] = (stock_data['Volatility'] > stock_data['Volatility'].quantile(0.95)).astype(int)
        stock_data.dropna(inplace=True) # drop missing data
        data[ticker] = stock_data
    return data


# Function to calculate RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Sidebar inputs
st.sidebar.title("Stock Comparison Dashboard")
tickers = st.sidebar.multiselect(
    'Select stocks to compare',
    ['GOOGL', 'AAPL', 'MSFT', 'AMZN'],
    default=['GOOGL', 'AAPL', 'MSFT', 'AMZN']
)
start_date = st.sidebar.date_input('Start date', pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input('End date', pd.to_datetime('today'))

# Fetch data
data = fetch_data(tickers)

# Filter data by date
for ticker in data:
    data[ticker] = data[ticker].loc[start_date:end_date]

# Merge data into a single DataFrame for analysis
all_data = pd.concat(data.values())

# Display HTML content
st.markdown("<h1 style='text-align: center;'>Stock Performance</h1>", unsafe_allow_html=True)

# Visualization: Line chart for adjusted closing price
fig = px.line(all_data, x=all_data.index, y='Adj Close', color='Ticker', title='Adjusted Close Price Over Time')
st.plotly_chart(fig)

# Calculate key metrics: Percentage change, 52-week high/low, MACD, RSI
metrics = []
for ticker in tickers:
    df = data[ticker]
    df['52_week_high'] = df['Adj Close'].rolling(window=252).max()
    df['52_week_low'] = df['Adj Close'].rolling(window=252).min()

    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    window_length = 14
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    metrics.append(df)

# Combining metrics for comparison
metrics_df = pd.concat(metrics)
st.markdown("<h1 style='text-align: center;'>Key Metrics Comparison</h1>", unsafe_allow_html=True)
st.write(metrics_df)

# Visualization: MACD and RSI
st.markdown("<h1 style='text-align: center;'>Technical Indicators</h1>", unsafe_allow_html=True)

for ticker in tickers:
    df = data[ticker]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line'))
    fig.update_layout(title=f'MACD for {ticker}', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), mode='lines', name='Overbought', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), mode='lines', name='Oversold', line=dict(dash='dash')))
    fig.update_layout(title=f'RSI for {ticker}', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

# Visualization: 52-week high/low
st.markdown("<h1 style='text-align: center;'>52-Week High and Low</h1>", unsafe_allow_html=True)

fig = px.line(metrics_df, x=metrics_df.index, y='52_week_high', color='Ticker', title='52-Week High')
st.plotly_chart(fig)

fig = px.line(metrics_df, x=metrics_df.index, y='52_week_low', color='Ticker', title='52-Week Low')
st.plotly_chart(fig)

# Display performance metrics
performance_metrics = pd.DataFrame({
    'Ticker': tickers,
    '52-Week High': [data[ticker]['52_week_high'].max() for ticker in tickers],
    '52-Week Low': [data[ticker]['52_week_low'].min() for ticker in tickers],
    'MACD': [data[ticker]['MACD'].iloc[-1] for ticker in tickers],
    'RSI': [data[ticker]['RSI'].iloc[-1] for ticker in tickers]
})

# Visualization: Performance Metrics
st.markdown("<h1 style='text-align: center;'>Performance Metrics</h1>", unsafe_allow_html=True)
fig = px.bar(performance_metrics, x='Ticker', y=['52-Week High', '52-Week Low', 'MACD', 'RSI'],
             title='Performance Metrics')
st.plotly_chart(fig)

# Latest financial year analysis section
st.markdown("<h1 style='text-align: center;'>Latest Financial Year Analysis</h1>", unsafe_allow_html=True)

for ticker in tickers:
    df = data[ticker]
    last_financial_year = df.loc[pd.to_datetime('today') - pd.DateOffset(years=1):]
    st.header(f"Analysis for {ticker}")
    st.write(last_financial_year)


# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


csv = convert_df_to_csv(metrics_df)

# Add download button
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='stock_analysis.csv',
    mime='text/csv',
)

# Initialize News API
load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))


# Function to fetch news sentiment and articles
def fetch_news_sentiment(ticker):
    news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    articles_info = []
    positive_count = 0
    negative_count = 0

    for article in news['articles']:
        description = article['description']
        if description:  # Only analyze if the description is not None
            score = analyzer.polarity_scores(description)
            sentiment_scores.append(score['compound'])
            articles_info.append({
                'title': article['title'],
                'description': description,
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name'],
                'sentiment': 'positive' if score['compound'] >= 0 else 'negative'
            })
            if score['compound'] >= 0:
                positive_count += 1
            else:
                negative_count += 1

    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    else:
        average_sentiment = 0  # Default sentiment if no articles have descriptions

    return average_sentiment, articles_info, positive_count, negative_count


# Analyze news sentiment for each ticker
news_sentiments = {}
articles_dict = {}
positive_counts = {}
negative_counts = {}
for ticker in tickers:
    sentiment, articles, positive_count, negative_count = fetch_news_sentiment(ticker)
    news_sentiments[ticker] = sentiment
    articles_dict[ticker] = articles
    positive_counts[ticker] = positive_count
    negative_counts[ticker] = negative_count

# Convert the sentiments into a DataFrame for visualization
sentiment_df = pd.DataFrame(list(news_sentiments.items()), columns=['Ticker', 'Sentiment'])

# Plot the average sentiments
st.markdown("<h1 style='text-align: center;'>News Sentiment Analysis</h1>", unsafe_allow_html=True)
fig = px.bar(sentiment_df, x='Ticker', y='Sentiment', title='Average Sentiment for Each Ticker')
st.plotly_chart(fig)

# Plot the positive and negative news counts
positive_negative_df = pd.DataFrame({
    'Ticker': list(positive_counts.keys()),
    'Positive': list(positive_counts.values()),
    'Negative': list(negative_counts.values())
})
positive_negative_df = positive_negative_df.melt(id_vars=['Ticker'], value_vars=['Positive', 'Negative'],
                                                 var_name='Sentiment', value_name='Count')

fig2 = px.bar(positive_negative_df, x='Ticker', y='Count', color='Sentiment', barmode='group',
              title='Positive and Negative News Counts for Each Ticker')
st.plotly_chart(fig2)

# Display news articles
st.markdown("<h1 style='text-align: center;'>News Articles</h1>", unsafe_allow_html=True)
for ticker in tickers:
    st.header(f"News for {ticker}")
    articles = articles_dict[ticker]

    # Display at least 5 articles
    for i, article in enumerate(articles[:5]):
        st.markdown(f"**{article['title']}**")
        st.markdown(f"*Source: {article['source']}*")
        st.markdown(
            f"Published at: {datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"{article['description']}")
        st.markdown(f"[Read more]({article['url']})")
        st.markdown("---")

    # Add a slider for remaining articles
    if len(articles) > 5:
        article_slider = st.slider(f'See more articles for {ticker}', 6, len(articles), 6)
        for i in range(5, article_slider):
            if i < len(articles):
                article = articles[i]
                st.markdown(f"**{article['title']}**")
                st.markdown(f"*Source: {article['source']}*")
                st.markdown(
                    f"Published at: {datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"{article['description']}")
                st.markdown(f"[Read more]({article['url']})")
                st.markdown("---")

