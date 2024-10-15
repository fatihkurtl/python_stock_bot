import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data = self.stock.history(period="1y")
        
    def technical_analysis(self):
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
    def fundamental_analysis(self):
        info = self.stock.info
        self.pe_ratio = info.get('trailingPE', None)
        self.pb_ratio = info.get('priceToBook', None)
        self.dividend_yield = info.get('dividendYield', None)
        self.market_cap = info.get('marketCap', None)
        
    def predict_price(self):
        df = self.data.copy()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD']
        X = df[features]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        last_data = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_data)[0]
        
        return prediction
        
    def analyze(self):
        self.technical_analysis()
        self.fundamental_analysis()
        prediction = self.predict_price()
        
        current_price = self.data['Close'].iloc[-1]
        
        print(f"Analysis for {self.ticker}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        print(f"P/E Ratio: {self.pe_ratio:.2f}" if self.pe_ratio else "P/E Ratio: N/A")
        print(f"P/B Ratio: {self.pb_ratio:.2f}" if self.pb_ratio else "P/B Ratio: N/A")
        print(f"Dividend Yield: {self.dividend_yield:.2%}" if self.dividend_yield else "Dividend Yield: N/A")
        print(f"Market Cap: ${self.market_cap:,.0f}" if self.market_cap else "Market Cap: N/A")
        
        last_data = self.data.iloc[-1]
        if last_data['Close'] > last_data['MA50'] > last_data['MA200']:
            print("Technical Indicator: Bullish (Price above MA50 and MA200)")
        elif last_data['Close'] < last_data['MA50'] < last_data['MA200']:
            print("Technical Indicator: Bearish (Price below MA50 and MA200)")
        else:
            print("Technical Indicator: Neutral")
        
        if prediction > current_price:
            print(f"Prediction: Stock may rise by {((prediction/current_price)-1)*100:.2f}%")
        else:
            print(f"Prediction: Stock may fall by {((current_price/prediction)-1)*100:.2f}%")
        
        self.plot_data()
        
    def plot_data(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.plot(self.data.index, self.data['MA50'], label='50-day MA')
        plt.plot(self.data.index, self.data['MA200'], label='200-day MA')
        plt.title(f'{self.ticker} Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    while True:
        ticker = input("Enter a NASDAQ stock ticker (or 'quit' to exit): ").upper()
        if ticker == 'QUIT':
            break
        try:
            analyzer = StockAnalyzer(ticker)
            analyzer.analyze()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()