Stock Performance and News Sentiment Analysis Dashboard
This project is a Stock Performance and News Sentiment Analysis Dashboard built using Streamlit. The dashboard allows users to compare various stocks based on key metrics, technical indicators, and news sentiment. The application fetches stock data using yfinance and news articles using the News API.

Features
Stock Comparison: Compare multiple stocks based on various financial metrics.
Technical Indicators: Display key technical indicators such as MACD and RSI.
52-Week High/Low: Visualize the 52-week high and low prices for selected stocks.
News Sentiment Analysis: Analyze the sentiment of recent news articles for each stock.
Download Data: Download the analyzed data as a CSV file.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/omoroclive/stock-dashboard.git
cd stock-dashboard
Create and activate a virtual environment:

bash
Copy code
python3 -m venv 
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up environment variables:

Create a .env file in the root directory of the project.

Add your News API key to the .env file:

env
Copy code
NEWS_API_KEY=NEWS_API_KEY
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the dashboard:

The dashboard will be accessible at http://localhost:8501.
Project Structure
bash
Copy code
.
├── app.py
├── requirements.txt
├── .env
└── README.md
app.py: The main script that contains the Streamlit application code.
requirements.txt: The list of dependencies required to run the application.
.env: File to store environment variables (not included in the repository for security reasons).
README.md: Project documentation.
Code Overview
Fetching and Preparing Stock Data
The fetch_data function uses yfinance to download stock data for the selected tickers. It calculates additional metrics such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and volatility.

python
Copy code
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
        stock_data.dropna(inplace=True)  # Drop missing data
        data[ticker] = stock_data
    return data
News Sentiment Analysis
The fetch_news_sentiment function fetches news articles for a given ticker using the News API and analyzes the sentiment of each article using VADER Sentiment Analysis.

python
Copy code
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
Dashboard Layout
The dashboard is organized into several sections:

Sidebar: Users can select the stocks to compare and the date range for analysis.
Stock Performance: Line chart displaying the adjusted closing prices over time.
Technical Indicators: Charts displaying the MACD and RSI for each stock.
52-Week High/Low: Line charts for the 52-week high and low prices.
News Sentiment Analysis: Bar charts for average sentiment and positive/negative news counts, along with a list of news articles.
Latest Financial Year Analysis
This section displays the stock data for the latest financial year for each selected ticker.

python
Copy code
st.markdown("<h1 style='text-align: center;'>Latest Financial Year Analysis</h1>", unsafe_allow_html=True)

for ticker in tickers:
    df = data[ticker]
    last_financial_year = df.loc[pd.to_datetime('today') - pd.DateOffset(years=1):]
    st.header(f"Analysis for {ticker}")
    st.write(last_financial_year)
Download Data as CSV
Users can download the analyzed data as a CSV file.

python
Copy code
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(metrics_df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='stock_analysis.csv',
    mime='text/csv',
)
Contribution
Feel free to contribute to this project by opening issues or submitting pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

