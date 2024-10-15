import sys
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QTextEdit, QLineEdit, QSplitter
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data = self.stock.history(period="1y")
        
    def technical_analysis(self):
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
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
        self.revenue = info.get('totalRevenue', None)
        self.profit_margin = info.get('profitMargins', None)
        
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
        
        analysis_result = f"Detaylı Analiz için {self.ticker}:\n\n"
        analysis_result += f"Güncel Fiyat: ${current_price:.2f}\n"
        analysis_result += f"Tahmini Fiyat: ${prediction:.2f}\n\n"
        
        analysis_result += "Temel Analiz:\n"
        analysis_result += f"P/E Ratio: {self.pe_ratio:.2f}\n" if self.pe_ratio else "P/E Ratio: N/A\n"
        analysis_result += f"P/B Ratio: {self.pb_ratio:.2f}\n" if self.pb_ratio else "P/B Ratio: N/A\n"
        analysis_result += f"Temettü Verimi: {self.dividend_yield:.2%}\n" if self.dividend_yield else "Temettü Verimi: N/A\n"
        analysis_result += f"Piyasa Değeri: ${self.market_cap:,.0f}\n" if self.market_cap else "Piyasa Değeri: N/A\n"
        analysis_result += f"Gelir: ${self.revenue:,.0f}\n" if self.revenue else "Gelir: N/A\n"
        analysis_result += f"Kâr Marjı: {self.profit_margin:.2%}\n\n" if self.profit_margin else "Kâr Marjı: N/A\n\n"
        
        analysis_result += "Teknik Analiz:\n"
        last_data = self.data.iloc[-1]
        if last_data['Close'] > last_data['MA50'] > last_data['MA200']:
            analysis_result += "Boğa (MA50 ve MA200'ün üzerinde fiyat)\n"
        elif last_data['Close'] < last_data['MA50'] < last_data['MA200']:
            analysis_result += "Teknik Gösterge: Ayı (Fiyat MA50 ve MA200'ün altında)\n"
        else:
            analysis_result += "Technical Indicator: Neutral\n"
        
        analysis_result += f"RSI: {last_data['RSI']:.2f}\n"
        analysis_result += f"MACD: {last_data['MACD']:.2f}\n"
        analysis_result += f"Signal Line: {last_data['Signal Line']:.2f}\n\n"
        
        if prediction > current_price:
            analysis_result += f"Tahmin:Hisse senedi şu kadar yükselebilir {((prediction/current_price)-1)*100:.2f}%\n"
            analysis_result += "Öneri:Hisseyi satın almayı veya elde tutmayı düşünün.\n"
        else:
            analysis_result += f"Tahmin:Hisse senedi şu kadar düşebilir {((current_price/prediction)-1)*100:.2f}%\n"
            analysis_result += "Öneri:Hisse senedini satmayı veya açığa satmayı düşünün.\n"
        
        analysis_result += "\nNot: Bu analiz geçmiş verilere ve tahminlere dayanmaktadır. Yatırım kararları vermeden önce her zaman kendi araştırmanızı yapın ve bir mali danışmana danışın."
        
        return analysis_result

    def plot_data(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.data.index, self.data['Close'], label='Close Price')
        ax.plot(self.data.index, self.data['MA50'], label='50 günlük MA')
        ax.plot(self.data.index, self.data['MA200'], label='2200 günlük MA')
        ax.set_title(f'{self.ticker} Hisse Senedi Fiyatı ve Hareketli Ortalamalar')
        ax.set_xlabel('Tarih')
        ax.set_ylabel('Fiyat')
        ax.legend()
        return fig

class StockAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gelişmiş Hisse Senedi Analizörü")
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left side - Stock list and search
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Hisse senedi arama...")
        self.search_bar.textChanged.connect(self.filter_stocks)
        
        self.stock_list = QListWidget()
        self.load_nasdaq_stocks()
        self.stock_list.itemClicked.connect(self.analyze_stock)
        
        left_layout.addWidget(QLabel("NASDAQ Hisse Senetleri:"))
        left_layout.addWidget(self.search_bar)
        left_layout.addWidget(self.stock_list)
        
        left_widget.setLayout(left_layout)
        
        # Sağ taraf - Analiz sonuçları ve grafik
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Kurye", 10))
        
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout()
        self.graph_widget.setLayout(self.graph_layout)
        
        right_layout.addWidget(QLabel("Analiz Sonuçları:"))
        right_layout.addWidget(self.result_text)
        right_layout.addWidget(self.graph_widget)
        
        right_widget.setLayout(right_layout)
        
        # Ana düzene widget ekleme
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Gerçek zamanlı güncellemeler için zamanlayıcı
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_analysis)
        self.timer.start(60000)  # Her 60 saniyede bir güncelleme
        
    def load_nasdaq_stocks(self):
        try:
            nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(nasdaq_url)

            for table in tables:
            # Genellikle ilk tablo NASDAQ 100 listesini içerir
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_column = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    nasdaq_stocks = table[ticker_column].tolist()
                    self.stock_list.addItems(nasdaq_stocks)
                    return

            # default hisseler, hisselere erisilemediyse gelir
            print("Uyarı: Wikipedia'dan NASDAQ hisse senetleri yüklenemedi. Varsayılan liste kullanılıyor.")
            default_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX", "LOGI", "CRM", "INTC", "CSCO"]
            self.stock_list.addItems(default_stocks)

        except Exception as e:
            print(f"NASDAQ hisse senetleri yüklenirken hata oluştu: {str(e)}")
            default_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]
            self.stock_list.addItems(default_stocks)

        
    def filter_stocks(self):
        search_text = self.search_bar.text().lower()
        for i in range(self.stock_list.count()):
            item = self.stock_list.item(i)
            if search_text in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def analyze_stock(self, item):
        ticker = item.text()
        try:
            analyzer = StockAnalyzer(ticker)
            result = analyzer.analyze()
            self.result_text.setText(result)
            
            # Önceki grafiği temizler
            for i in reversed(range(self.graph_layout.count())): 
                self.graph_layout.itemAt(i).widget().setParent(None)
            
            # Yeni grafik ekler
            fig = analyzer.plot_data()
            canvas = FigureCanvas(fig)
            self.graph_layout.addWidget(canvas)
            
        except Exception as e:
            self.result_text.setText(f"Bir hata oluştu: {str(e)}\nLütfen hisse senedi sembolünü kontrol edin ve tekrar deneyin.")
    
    def update_analysis(self):
        current_item = self.stock_list.currentItem()
        if current_item:
            self.analyze_stock(current_item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalyzerApp()
    window.show()
    sys.exit(app.exec())