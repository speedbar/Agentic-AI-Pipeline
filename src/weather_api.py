import requests
from config import WeatherConfig
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WeatherAPI:
    def __init__(self):
        self.api_key = WeatherConfig.api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        # \"\"\"Fetch current weather for a location\"\"\"
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                "location": data.get("name"),
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
            }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"error": str(e)}
    
    def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        # \"\"\"Fetch weather forecast\"\"\"
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Forecast API error: {e}")
            return {"error": str(e)}