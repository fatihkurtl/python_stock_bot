import sys
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QPushButton, QLabel, QTextEdit, QLineEdit, 
                             QSplitter, QTabWidget, QProgressBar, QMessageBox, QComboBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
from bs4 import BeautifulSoup

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
        
        self.data['ATR'] = self.calculate_atr(14)
        
    def calculate_atr(self, period):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
        
    def fundamental_analysis(self):
        info = self.stock.info
        self.pe_ratio = info.get('trailingPE', None)
        self.pb_ratio = info.get('priceToBook', None)
        self.dividend_yield = info.get('dividendYield', None)
        self.market_cap = info.get('marketCap', None)
        self.revenue = info.get('totalRevenue', None)
        self.profit_margin = info.get('profitMargins', None)
        self.debt_to_equity = info.get('debtToEquity', None)
        self.current_ratio = info.get('currentRatio', None)
        self.quick_ratio = info.get('quickRatio', None)
        
    def predict_price(self):
        df = self.data.copy()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD', 'ATR']
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
        
        analysis_result = f"Detaylı Analiz: {self.ticker}\n\n"
        analysis_result += f"Güncel Fiyat: ${current_price:.2f}\n"
        analysis_result += f"Tahmini Fiyat: ${prediction:.2f}\n\n"
        
        analysis_result += "Temel Analiz:\n"
        analysis_result += f"P/E Oranı: {self.pe_ratio:.2f}\n" if self.pe_ratio else "P/E Oranı: N/A\n"
        analysis_result += f"P/B Oranı: {self.pb_ratio:.2f}\n" if self.pb_ratio else "P/B Oranı: N/A\n"
        analysis_result += f"Temettü Verimi: {self.dividend_yield:.2%}\n" if self.dividend_yield else "Temettü Verimi: N/A\n"
        analysis_result += f"Piyasa Değeri: ${self.market_cap:,.0f}\n" if self.market_cap else "Piyasa Değeri: N/A\n"
        analysis_result += f"Gelir: ${self.revenue:,.0f}\n" if self.revenue else "Gelir: N/A\n"
        analysis_result += f"Kâr Marjı: {self.profit_margin:.2%}\n" if self.profit_margin else "Kâr Marjı: N/A\n"
        analysis_result += f"Borç/Özsermaye Oranı: {self.debt_to_equity:.2f}\n" if self.debt_to_equity else "Borç/Özsermaye Oranı: N/A\n"
        analysis_result += f"Cari Oran: {self.current_ratio:.2f}\n" if self.current_ratio else "Cari Oran: N/A\n"
        analysis_result += f"Asit-Test Oranı: {self.quick_ratio:.2f}\n\n" if self.quick_ratio else "Asit-Test Oranı: N/A\n\n"
        
        analysis_result += "Teknik Analiz:\n"
        last_data = self.data.iloc[-1]
        if last_data['Close'] > last_data['MA50'] > last_data['MA200']:
            analysis_result += "Trend: Yükseliş (Fiyat MA50 ve MA200'ün üzerinde)\n"
        elif last_data['Close'] < last_data['MA50'] < last_data['MA200']:
            analysis_result += "Trend: Düşüş (Fiyat MA50 ve MA200'ün altında)\n"
        else:
            analysis_result += "Trend: Yatay\n"
        
        analysis_result += f"RSI: {last_data['RSI']:.2f}\n"
        analysis_result += f"MACD: {last_data['MACD']:.2f}\n"
        analysis_result += f"Sinyal Hattı: {last_data['Signal Line']:.2f}\n"
        analysis_result += f"ATR: {last_data['ATR']:.2f}\n\n"
        
        if prediction > current_price:
            analysis_result += f"Tahmin: Hisse senedi şu kadar yükselebilir {((prediction/current_price)-1)*100:.2f}%\n"
            analysis_result += "Öneri: Hisseyi satın almayı veya elde tutmayı düşünün.\n"
        else:
            analysis_result += f"Tahmin: Hisse senedi şu kadar düşebilir {((current_price/prediction)-1)*100:.2f}%\n"
            analysis_result += "Öneri: Hisse senedini satmayı veya açığa satmayı düşünün.\n"
        
        analysis_result += "\nNot: Bu analiz geçmiş verilere ve tahminlere dayanmaktadır. Yatırım kararları vermeden önce her zaman kendi araştırmanızı yapın ve bir mali danışmana danışın."
        
        return analysis_result

    def plot_data(self):
        fig = Figure(figsize=(10, 15))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
        # Fiyat ve Hareketli Ortalamalar
        ax1.plot(self.data.index, self.data['Close'], label='Kapanış Fiyatı')
        ax1.plot(self.data.index, self.data['MA50'], label='50 günlük MA')
        ax1.plot(self.data.index, self.data['MA200'], label='200 günlük MA')
        ax1.set_title(f'{self.ticker} Hisse Senedi Fiyatı ve Hareketli Ortalamalar')
        ax1.set_ylabel('Fiyat')
        ax1.legend()
        
        # RSI
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='red', linestyle='--')
        ax2.axhline(y=30, color='green', linestyle='--')
        ax2.set_title('Göreceli Güç Endeksi (RSI)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        
        # MACD
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data.index, self.data['Signal Line'], label='Sinyal Hattı', color='orange')
        ax3.bar(self.data.index, self.data['MACD'] - self.data['Signal Line'], label='MACD Histogramı', color='gray', alpha=0.3)
        ax3.set_title('MACD')
        ax3.set_xlabel('Tarih')
        ax3.set_ylabel('MACD')
        ax3.legend()
        
        fig.tight_layout()
        return fig

class StockLoaderThread(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)

    def run(self):
        try:
            # İlk olarak Wikipedia'dan çekmeyi deneyelim
            nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            response = requests.get(nasdaq_url)
            tables = pd.read_html(response.text)
            
            for table in tables:
                if 'Ticker' in table.columns:
                    nasdaq_stocks = table['Ticker'].tolist()
                    break
            else:
                raise ValueError("NASDAQ-100 hisseleri bulunamadı")

            if not nasdaq_stocks:
                raise ValueError("NASDAQ-100 hisseleri listesi boş")

        except Exception as e:
            print(f"Wikipedia'dan yükleme başarısız oldu: {str(e)}")
            print("Alternatif kaynaktan yükleniyor...")

            try:
                # Alternatif kaynak: NASDAQ'ın resmi web sitesi
                url = "https://api.nasdaq.com/api/quote/list-type/NASDAQ100"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers)
                data = response.json()
                nasdaq_stocks = [stock['symbol'] for stock in data['data']['data']['rows']]

            except Exception as e:
                print(f"Alternatif kaynaktan yükleme başarısız oldu: {str(e)}")
                print("Varsayılan hisse listesi kullanılıyor...")
                nasdaq_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]

        total_stocks = len(nasdaq_stocks)
        for i, stock in enumerate(nasdaq_stocks):
            self.progress.emit(int((i + 1) / total_stocks * 100))
        
        self.finished.emit(nasdaq_stocks)
        
        
#! Deneysel Alan
##############################################################################
# class StockLoaderThread(QThread):
#     finished = pyqtSignal(list)
#     progress = pyqtSignal(int)

#     def run(self):
#         try:
#             # İlk olarak Wikipedia'dan çekmeyi deneyelim
#             sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#             response = requests.get(sp500_url)
#             tables = pd.read_html(response.text)
            
#             sp500_stocks = []
#             for table in tables:
#                 if 'Symbol' in table.columns:
#                     sp500_stocks = table['Symbol'].tolist()
#                     break
#             else:
#                 raise ValueError("S&P 500 hisseleri bulunamadı")

#             if not sp500_stocks:
#                 raise ValueError("S&P 500 hisseleri listesi boş")

#         except Exception as e:
#             print(f"Wikipedia'dan yükleme başarısız oldu: {str(e)}")
#             print("Alternatif kaynaktan yükleniyor...")

#             try:
#                 # Alternatif kaynak: S&P 500 hisselerini saglayan API
#                 url = "https://api.example.com/sp500"
#                 response = requests.get(url)
#                 data = response.json()
#                 sp500_stocks = [stock['symbol'] for stock in data['data']]

#             except Exception as e:
#                 print(f"Alternatif kaynaktan yükleme başarısız oldu: {str(e)}")
#                 print("Varsayılan hisse listesi kullanılıyor...")
#                 sp500_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]

#         total_stocks = len(sp500_stocks)
#         for i, stock in enumerate(sp500_stocks):
#             self.progress.emit(int((i + 1) / total_stocks * 100))
        
#         self.finished.emit(sp500_stocks)
##############################################################################        


class StockAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gelişmiş NASDAQ-100 Hisse Senedi Analizörü")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("python_stock_bot\\assets\\favicon.ico"))
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Sol kenar - Hisse listesi ve arama
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Hisse senedi arama...")
        self.search_bar.textChanged.connect(self.filter_stocks)
        
        self.stock_list = QListWidget()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        left_layout.addWidget(QLabel("NASDAQ-100 Hisse Senetleri:"))
        left_layout.addWidget(self.search_bar)
        left_layout.addWidget(self.stock_list)
        left_layout.addWidget(self.progress_bar)
        
        left_widget.setLayout(left_layout)
        
        # Sağ kenar - Analiz sonuçları ve grafik
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        
        # Analiz sonuçları sekmesi
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Courier", 10))
        
        # Grafik sekmesi
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout()
        
        self.graph_widget.setLayout(self.graph_layout)
        
        # Karşılaştırma sekmesi
        
        self.comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()
        self.comparison_combo = QComboBox()
        self.comparison_result = QTableWidget()
        self.comparison_result.setColumnCount(3)
        self.comparison_result.setHorizontalHeaderLabels(["Metrik", "Seçili Hisse", "Karşılaştırılan Hisse"])
        self.comparison_result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        comparison_layout.addWidget(QLabel("Karşılaştırılacak Hisse Senedi:"))
        
        comparison_layout.addWidget(self.comparison_combo)
        comparison_layout.addWidget(self.comparison_result)
        self.comparison_widget.setLayout(comparison_layout)
        
        self.tab_widget.addTab(self.result_text, "Analiz Sonuçları")
        self.tab_widget.addTab(self.graph_widget, "Grafik")
        self.tab_widget.addTab(self.comparison_widget, "Karşılaştırma")
        
        right_layout.addWidget(self.tab_widget)
        
        right_widget.setLayout(right_layout)
        
        # Ana düzene widget ekleme
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Menü çubuğu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Dosya')
        
        exit_action = file_menu.addAction('Çıkış')
        exit_action.triggered.connect(self.close)
        
        help_menu = menubar.addMenu('Yardım')
        about_action = help_menu.addAction('Hakkında')
        about_action.triggered.connect(self.show_about)
        
        # Gerçek zamanlı güncellemeler için zamanlayıcı
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_analysis)
        self.timer.start(60000)  # Her 60 saniyede bir güncelleme
        
        self.load_nasdaq_stocks()
        
    def load_nasdaq_stocks(self):
        self.progress_bar.setVisible(True)
        self.stock_loader_thread = StockLoaderThread()
        self.stock_loader_thread.finished.connect(self.on_stock_load_finished)
        self.stock_loader_thread.progress.connect(self.progress_bar.setValue)
        self.stock_loader_thread.start()

    def on_stock_load_finished(self, stocks):
        self.stock_list.clear()
        self.stock_list.addItems(stocks)
        self.comparison_combo.addItems(stocks)
        self.progress_bar.setVisible(False)
        self.stock_list.itemClicked.connect(self.analyze_stock)
        
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
            
            self.tab_widget.setCurrentIndex(0)  # Analiz sonuçları sekmesine geç
            
            # Karşılaştırma yap
            self.compare_stocks(ticker, self.comparison_combo.currentText())
            
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Bir hata oluştu: {str(e)}\nLütfen hisse senedi sembolünü kontrol edin ve tekrar deneyin.")
    
    def compare_stocks(self, stock1, stock2):
        if stock1 != stock2:
            try:
                analyzer1 = StockAnalyzer(stock1)
                analyzer2 = StockAnalyzer(stock2)
                
                analyzer1.analyze()
                analyzer2.analyze()
                
                comparison_data = [
                    ("Güncel Fiyat", f"${analyzer1.data['Close'].iloc[-1]:.2f}", f"${analyzer2.data['Close'].iloc[-1]:.2f}"),
                    ("RSI", f"{analyzer1.data['RSI'].iloc[-1]:.2f}", f"{analyzer2.data['RSI'].iloc[-1]:.2f}"),
                    ("MACD", f"{analyzer1.data['MACD'].iloc[-1]:.2f}", f"{analyzer2.data['MACD'].iloc[-1]:.2f}"),
                    ("P/E Oranı", f"{analyzer1.pe_ratio:.2f}" if analyzer1.pe_ratio else "N/A", f"{analyzer2.pe_ratio:.2f}" if analyzer2.pe_ratio else "N/A"),
                    ("Temettü Verimi", f"{analyzer1.dividend_yield:.2%}" if analyzer1.dividend_yield else "N/A", f"{analyzer2.dividend_yield:.2%}" if analyzer2.dividend_yield else "N/A"),
                    ("Piyasa Değeri", f"${analyzer1.market_cap:,.0f}" if analyzer1.market_cap else "N/A", f"${analyzer2.market_cap:,.0f}" if analyzer2.market_cap else "N/A"),
                ]
                
                self.comparison_result.setRowCount(len(comparison_data))
                for i, (metric, value1, value2) in enumerate(comparison_data):
                    self.comparison_result.setItem(i, 0, QTableWidgetItem(metric))
                    self.comparison_result.setItem(i, 1, QTableWidgetItem(value1))
                    self.comparison_result.setItem(i, 2, QTableWidgetItem(value2))
                
            except Exception as e:
                QMessageBox.warning(self, "Hata", f"Karşılaştırma yapılırken bir hata oluştu: {str(e)}")
    
    def update_analysis(self):
        current_item = self.stock_list.currentItem()
        if current_item:
            self.analyze_stock(current_item)
    
    def show_about(self):
        QMessageBox.about(self, "Hakkında", "Gelişmiş NASDAQ-100 Hisse Senedi Analizörü\n\nVersiyon 1.0\n\nGeliştirici: https://fatihkurt.web.tr\n\n© 2024 Tüm hakları saklıdır.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Uygulama stili
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = StockAnalyzerApp()
    window.show()
    sys.exit(app.exec())