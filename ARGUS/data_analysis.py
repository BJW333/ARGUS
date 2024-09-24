import requests
import pandas as pd
import schedule
import time
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup

class DataStream:
    """Base class for all data streams."""

    def fetch_data(self):
        """Fetch data from the source."""
        raise NotImplementedError("Subclasses should implement this method.")

    def analyze_data(self, data):
        """Analyze fetched data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def visualize_data(self, data):
        """Visualize data (optional)."""
        pass  # Optional implementation in subclasses

class StockDataStream(DataStream):
    """Fetch and analyze stock data from Alpha Vantage API."""

    def __init__(self, symbol, api_key):
        self.symbol = symbol
        self.api_key = api_key
        self.endpoint = 'https://www.alphavantage.co/query'

    def fetch_data(self):
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.symbol,
            'interval': '1min',
            'apikey': self.api_key
        }
        response = requests.get(self.endpoint, params=params)
        data = response.json()

        if 'Time Series (1min)' in data:
            time_series = data['Time Series (1min)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df
        else:
            print(f"Error fetching stock data for {self.symbol}: {data.get('Note', 'No additional information')}")
            return pd.DataFrame()

    def analyze_data(self, data):
        if data.empty:
            print(f"No data available for stock {self.symbol}.")
            return

        latest_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        price_change = latest_price - previous_price
        percent_change = (price_change / previous_price) * 100

        print(f"Stock {self.symbol}: Latest Price ${latest_price:.2f}")

        if abs(percent_change) > 1:  # Threshold for significant change
            print(f"Significant price change detected: {percent_change:.2f}%")
        else:
            print(f"Minor price fluctuation: {percent_change:.2f}%")

    def visualize_data(self, data):
        if data.empty:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], label=f'{self.symbol} Close Price')
        plt.title(f'{self.symbol} Stock Price')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
class FlightDataStream(DataStream):
    """Fetch and analyze flight data from AviationStack API."""

    def __init__(self, flight_number, api_key):
        self.flight_number = flight_number
        self.api_key = api_key
        self.endpoint = 'http://api.aviationstack.com/v1/flights'

    def fetch_data(self):
        params = {
            'access_key': self.api_key,
            'flight_iata': self.flight_number
        }
        response = requests.get(self.endpoint, params=params)
        flight_data = response.json()

        if 'data' in flight_data and flight_data['data']:
            flight_info = flight_data['data'][0]
            return flight_info
        else:
            print(f"Error fetching flight data for {self.flight_number}: {flight_data.get('error', {}).get('info', 'No additional information')}")
            return {}

    def analyze_data(self, data):
        if not data:
            print(f"No data available for flight {self.flight_number}.")
            return

        flight_status = data['flight_status']
        departure_airport = data['departure']['airport']
        arrival_airport = data['arrival']['airport']
        departure_time = data['departure']['estimated']
        arrival_time = data['arrival']['estimated']

        print(f"Flight {self.flight_number}:")
        print(f"  Status: {flight_status}")
        print(f"  Departure: {departure_airport} at {departure_time}")
        print(f"  Arrival: {arrival_airport} at {arrival_time}")

        if flight_status == "delayed":
            print(f"Flight {self.flight_number} is delayed.")
        else:
            print(f"Flight {self.flight_number} is on schedule.")

    def visualize_data(self, data):
        # Optional visualization for flight status, could implement plotting here
        pass
    
class WeatherDataStream(DataStream):
    """Fetch and analyze weather data from Open-Meteo API."""
    def __init__(self, city, latitude, longitude):
        self.city = city
        self.latitude = latitude
        self.longitude = longitude
        self.endpoint = 'https://api.open-meteo.com/v1/forecast'

    def fetch_data(self):
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'current_weather': 'true',
        }
        response = requests.get(self.endpoint, params=params)
        data = response.json()

        if 'current_weather' in data:
            weather_info = {
                'City': self.city,
                'Temperature': data['current_weather']['temperature'],
                'Wind Speed': data['current_weather']['windspeed'],
                'Weather': data['current_weather']['weathercode'],
            }
            return weather_info
        else:
            print(f"Error fetching weather data for {self.city}: {data}")
            return {}

    def analyze_data(self, data):
        if not data:
            print(f"No weather data available for {self.city}.")
            return

        print(f"Weather in {data['City']}:")
        print(f"  Temperature: {data['Temperature']}°C")
        print(f"  Wind Speed: {data['Wind Speed']} km/h")
        print(f"  Weather Code: {data['Weather']} (refer to Open-Meteo documentation for interpretation)")

    
class CryptoDataStream(DataStream):
    """Fetch and analyze cryptocurrency data from CoinGecko API."""

    def __init__(self, coin_id='bitcoin', vs_currency='usd'):
        self.coin_id = coin_id
        self.vs_currency = vs_currency
        self.endpoint = 'https://api.coingecko.com/api/v3/simple/price'
        self.change_endpoint = 'https://api.coingecko.com/api/v3/coins/markets'

    def fetch_data(self):
        # Fetch current price
        params = {
            'ids': self.coin_id,
            'vs_currencies': self.vs_currency,
        }
        response = requests.get(self.endpoint, params=params)
        price_data = response.json()

        # Fetch price change percentage
        change_params = {
            'vs_currency': self.vs_currency,
            'ids': self.coin_id,
            'order': 'market_cap_desc',
            'per_page': 1,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d'
        }
        change_response = requests.get(self.change_endpoint, params=change_params)
        change_data = change_response.json()

        if price_data and change_data:
            current_price = price_data[self.coin_id][self.vs_currency]
            price_change_24h = change_data[0].get('price_change_percentage_24h_in_currency', 0)
            data = {
                'name': change_data[0]['name'],
                'symbol': change_data[0]['symbol'],
                'current_price': current_price,
                'price_change_24h': price_change_24h
            }
            return data
        else:
            print(f"Error fetching data for {self.coin_id}")
            return {}

    def analyze_data(self, data):
        if not data:
            print(f"No data available for cryptocurrency {self.coin_id}.")
            return

        current_price = data.get('current_price')
        price_change_24h = data.get('price_change_24h')
        print(f"Cryptocurrency {data['name']} ({data['symbol'].upper()}):")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  24h Change: {price_change_24h:.2f}%")

        # Threshold for significant change
        if abs(price_change_24h) > 5:
            print("Significant price movement detected in the last 24 hours!")
        else:
            print("Price movement is within normal range.")

    def visualize_data(self, data):
        # Implement visualization if needed
        pass

class DataAnalysisFramework:
    """Real-time data analysis framework for ARGUS."""

    def __init__(self):
        self.data_streams = []

    def add_data_stream(self, data_stream):
        """Add a new data stream to the framework."""
        self.data_streams.append(data_stream)

    def run_analysis(self):
        """Run analysis on all added data streams."""
        print(f"\n--- Real-Time Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        for stream in self.data_streams:
            data = stream.fetch_data()
            stream.analyze_data(data)
            stream.visualize_data(data)

# # Replace with your actual API keys
# ALPHA_VANTAGE_API_KEY = 'MJX8BVSA9W1WOEH4'
# AVIATIONSTACK_API_KEY = '6087a4f837f7de0c30af184c6e886a9b'

# # Initialize the framework
# framework = DataAnalysisFramework()

# # Add data streams
# stock_stream = StockDataStream(symbol='AAPL', api_key=ALPHA_VANTAGE_API_KEY)
# weather_stream = WeatherDataStream(city='New York', latitude=40.7128, longitude=-74.0060)
# news_stream = NewsDataStream(url='https://www.bbc.com/news')
# crypto_stream = CryptoDataStream(coin_id='bitcoin', vs_currency='usd')
# flight_stream = FlightDataStream(flight_number='AA100', api_key=AVIATIONSTACK_API_KEY)

# framework.add_data_stream(stock_stream)
# framework.add_data_stream(weather_stream)
# framework.add_data_stream(news_stream)
# framework.add_data_stream(crypto_stream)
# framework.add_data_stream(flight_stream)

# # Schedule the analysis to run every 5 minutes
# schedule.every(5).minutes.do(framework.run_analysis)

# # Run the initial analysis
# framework.run_analysis()

# print("Starting real-time data analysis framework for ARGUS...")

# # Keep the script running and execute scheduled tasks
# while True:
#     schedule.run_pending()
#     time.sleep(1)
