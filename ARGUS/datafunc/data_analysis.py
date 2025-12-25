from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


#CORE ABSTRACTIONS

@dataclass
class StreamResult:
    """Unified return type for all data streams."""
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get(self, *keys, default=None):
        """Safe nested access: result.get('departure', 'airport')"""
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val


class DataStream(ABC):
    """Base class with built-in resilience."""
    
    TIMEOUT = 10
    CACHE_TTL = timedelta(minutes=5)
    
    _session: Optional[requests.Session] = None
    _cache: dict[str, tuple[datetime, StreamResult]] = {}
    
    @classmethod
    def get_session(cls) -> requests.Session:
        """Shared session with retry logic."""
        if cls._session is None:
            cls._session = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retries)
            cls._session.mount('https://', adapter)
            cls._session.mount('http://', adapter)
        return cls._session
    
    def _request(self, url: str, params: dict, cache_key: str = None) -> StreamResult:
        """Make HTTP request with caching and error handling."""
        #check cache
        if cache_key and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if datetime.now() - cached_time < self.CACHE_TTL:
                cached_result.cached = True
                return cached_result
        
        try:
            resp = self.get_session().get(url, params=params, timeout=self.TIMEOUT)
            resp.raise_for_status()
            result = StreamResult(success=True, data=resp.json())
            
            if cache_key:
                self._cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except requests.Timeout:
            log.warning(f"Timeout fetching {url}")
            return StreamResult(success=False, error="Request timed out")
        except requests.HTTPError as e:
            log.warning(f"HTTP error {e.response.status_code}: {url}")
            return StreamResult(success=False, error=f"HTTP {e.response.status_code}")
        except requests.RequestException as e:
            log.error(f"Request failed: {e}")
            return StreamResult(success=False, error=str(e))
    
    @abstractmethod
    def fetch(self) -> StreamResult:
        """Fetch data from source."""
        pass
    
    @abstractmethod
    def analyze(self, result: StreamResult) -> dict[str, Any]:
        """Analyze data and return metrics dict (no printing)."""
        pass


#DATA STREAMS

class StockStream(DataStream):
    ENDPOINT = 'https://www.alphavantage.co/query'
    
    def __init__(self, symbol: str, api_key: str):
        self.symbol = symbol.upper()
        self.api_key = api_key
    
    def fetch(self) -> StreamResult:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': self.symbol,
            'interval': '5min',
            'apikey': self.api_key
        }
        result = self._request(self.ENDPOINT, params, cache_key=f"stock:{self.symbol}")
        
        if not result.success:
            return result
        
        ts = result.data.get('Time Series (5min)')
        if not ts:
            note = result.data.get('Note', result.data.get('Error Message', 'Unknown error'))
            return StreamResult(success=False, error=note)
        
        #parse into clean format
        prices = []
        for timestamp, vals in sorted(ts.items()):
            prices.append({
                'time': timestamp,
                'open': float(vals['1. open']),
                'high': float(vals['2. high']),
                'low': float(vals['3. low']),
                'close': float(vals['4. close']),
                'volume': int(vals['5. volume'])
            })
        
        return StreamResult(success=True, data={
            'symbol': self.symbol,
            'prices': prices,
            'latest': prices[-1] if prices else None
        })
    
    def analyze(self, result: StreamResult) -> dict[str, Any]:
        if not result.success or not result.data.get('prices'):
            return {'error': result.error or 'No data'}
        
        prices = result.data['prices']
        latest = prices[-1]['close']
        previous = prices[-2]['close'] if len(prices) > 1 else latest
        
        change = latest - previous
        pct_change = (change / previous) * 100 if previous else 0
        
        return {
            'symbol': self.symbol,
            'price': latest,
            'change': change,
            'percent_change': pct_change,
            'significant': abs(pct_change) > 1
        }


class WeatherStream(DataStream):
    ENDPOINT = 'https://api.open-meteo.com/v1/forecast'
    
    WEATHER_CODES = {
        0: 'clear skies', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
        45: 'foggy', 48: 'foggy',
        51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle',
        56: 'freezing drizzle', 57: 'freezing drizzle',
        61: 'light rain', 63: 'rain', 65: 'heavy rain',
        66: 'freezing rain', 67: 'freezing rain',
        71: 'light snow', 73: 'snow', 75: 'heavy snow',
        77: 'snow grains',
        80: 'light showers', 81: 'showers', 82: 'heavy showers',
        85: 'light snow showers', 86: 'snow showers',
        95: 'thunderstorms', 96: 'thunderstorms with hail', 99: 'severe thunderstorms'
    }
    
    def __init__(self, city: str, lat: float, lon: float):
        self.city = city
        self.lat = lat
        self.lon = lon
    
    def fetch(self, days: int = 1) -> StreamResult:
        """
        Fetch weather data.
        days=1: current only
        days=2-7: includes forecast
        """
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'current_weather': 'true',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode',
            'temperature_unit': 'fahrenheit',
            'windspeed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': 'auto',
            'forecast_days': min(days, 7)
        }
        
        result = self._request(self.ENDPOINT, params, cache_key=f"weather:{self.lat},{self.lon}:{days}")
        
        if not result.success:
            return result
        
        data = result.data
        current = data.get('current_weather', {})
        daily = data.get('daily', {})
        
        #parse current conditions
        code = current.get('weathercode', 0)
        current_data = {
            'temp_f': current.get('temperature'),
            'wind_mph': current.get('windspeed'),
            'condition_code': code,
            'condition': self.WEATHER_CODES.get(code, 'unknown')
        }
        
        #parse daily forecast
        forecast = []
        dates = daily.get('time', [])
        highs = daily.get('temperature_2m_max', [])
        lows = daily.get('temperature_2m_min', [])
        precip = daily.get('precipitation_sum', [])
        codes = daily.get('weathercode', [])
        
        for i in range(len(dates)):
            forecast.append({
                'date': dates[i],
                'high_f': highs[i] if i < len(highs) else None,
                'low_f': lows[i] if i < len(lows) else None,
                'precip_in': precip[i] if i < len(precip) else 0,
                'condition_code': codes[i] if i < len(codes) else 0,
                'condition': self.WEATHER_CODES.get(codes[i] if i < len(codes) else 0, 'unknown')
            })
        
        return StreamResult(success=True, data={
            'city': self.city,
            'current': current_data,
            'forecast': forecast
        })
    
    def analyze(self, result: StreamResult, day_offset: int = 0) -> dict:
        """
        Analyze weather for a specific day.
        day_offset=0: today/current
        day_offset=1: tomorrow
        day_offset=2: day after tomorrow, etc.
        """
        if not result.success:
            return {'error': result.error}
        
        data = result.data
        forecast = data.get('forecast', [])
        
        if day_offset == 0:
            #current weather
            current = data['current']
            return {
                'city': data['city'],
                'is_forecast': False,
                'temp_f': round(current['temp_f']) if current['temp_f'] else None,
                'condition': current['condition'],
                'wind_mph': round(current['wind_mph']) if current['wind_mph'] else None,
                'high_f': forecast[0]['high_f'] if forecast else None,
                'low_f': forecast[0]['low_f'] if forecast else None
            }
        else:
            #forecast for future day
            if day_offset >= len(forecast):
                return {'error': f'Forecast not available for {day_offset} days out'}
            
            day = forecast[day_offset]
            return {
                'city': data['city'],
                'is_forecast': True,
                'date': day['date'],
                'high_f': round(day['high_f']) if day['high_f'] else None,
                'low_f': round(day['low_f']) if day['low_f'] else None,
                'condition': day['condition'],
                'precip_in': day['precip_in']
            }


class FlightStream(DataStream):
    ENDPOINT = 'http://api.aviationstack.com/v1/flights'
    
    def __init__(self, flight_number: str, api_key: str):
        self.flight_number = flight_number.upper()
        self.api_key = api_key
    
    def fetch(self) -> StreamResult:
        params = {
            'access_key': self.api_key,
            'flight_iata': self.flight_number
        }
        result = self._request(self.ENDPOINT, params, cache_key=f"flight:{self.flight_number}")
        
        if not result.success:
            return result
        
        flights = result.data.get('data', [])
        if not flights:
            err = result.get('error', 'info') or 'Flight not found'
            return StreamResult(success=False, error=err)
        
        f = flights[0]
        return StreamResult(success=True, data={
            'flight': self.flight_number,
            'status': f.get('flight_status', 'unknown'),
            'departure': {
                'airport': f.get('departure', {}).get('airport'),
                'scheduled': f.get('departure', {}).get('scheduled'),
                'estimated': f.get('departure', {}).get('estimated'),
            },
            'arrival': {
                'airport': f.get('arrival', {}).get('airport'),
                'scheduled': f.get('arrival', {}).get('scheduled'),
                'estimated': f.get('arrival', {}).get('estimated'),
            }
        })
    
    def analyze(self, result: StreamResult) -> dict[str, Any]:
        if not result.success:
            return {'error': result.error}
        
        d = result.data
        return {
            'flight': d['flight'],
            'status': d['status'],
            'from': d['departure']['airport'],
            'to': d['arrival']['airport'],
            'departure_time': d['departure']['estimated'] or d['departure']['scheduled'],
            'arrival_time': d['arrival']['estimated'] or d['arrival']['scheduled'],
            'delayed': d['status'] == 'delayed'
        }


class CryptoStream(DataStream):
    ENDPOINT = 'https://api.coingecko.com/api/v3/coins/markets'
    
    def __init__(self, coin_id: str = 'bitcoin', vs_currency: str = 'usd'):
        self.coin_id = coin_id.lower()
        self.vs_currency = vs_currency.lower()
    
    def fetch(self) -> StreamResult:
        params = {
            'vs_currency': self.vs_currency,
            'ids': self.coin_id,
            'sparkline': 'false',
            'price_change_percentage': '1h,24h,7d'
        }
        result = self._request(self.ENDPOINT, params, cache_key=f"crypto:{self.coin_id}")
        
        if not result.success:
            return result
        
        data = result.data
        if not isinstance(data, list) or not data:
            return StreamResult(success=False, error='Coin not found')
        
        c = data[0]
        return StreamResult(success=True, data={
            'id': c.get('id'),
            'name': c.get('name'),
            'symbol': c.get('symbol', '').upper(),
            'price': c.get('current_price'),
            'change_1h': c.get('price_change_percentage_1h_in_currency'),
            'change_24h': c.get('price_change_percentage_24h_in_currency'),
            'change_7d': c.get('price_change_percentage_7d_in_currency'),
            'market_cap': c.get('market_cap'),
            'volume_24h': c.get('total_volume')
        })
    
    def analyze(self, result: StreamResult) -> dict[str, Any]:
        if not result.success:
            return {'error': result.error}
        
        d = result.data
        change_24h = d.get('change_24h') or 0
        
        return {
            'name': d['name'],
            'symbol': d['symbol'],
            'price': d['price'],
            'change_24h': round(change_24h, 2),
            'significant': abs(change_24h) > 5
        }