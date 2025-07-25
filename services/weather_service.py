import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import random
import math

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for weather data and forecasting"""

    def __init__(self):
        self.current_weather = {}
        self.forecast_data = []
        self.location = "Gurugram, Haryana, IN"  # Default location
        self.last_update = None
        self.update_interval = 300  # 5 minutes

        # Initialize with mock data
        self._initialize_weather_data()
        logger.info("Weather service initialized")

    def _initialize_weather_data(self):
        """Initialize with mock weather data"""
        # Generate realistic weather data for current location
        current_time = datetime.utcnow()

        # Simulate seasonal weather for Gurugram (North India)
        month = current_time.month
        hour = current_time.hour

        # Base temperatures by season (Fahrenheit)
        seasonal_temps = {
            # Winter (Dec, Jan, Feb)
            12: {'day': 68, 'night': 45},
            1: {'day': 65, 'night': 40},
            2: {'day': 70, 'night': 48},
            # Spring (Mar, Apr, May)
            3: {'day': 78, 'night': 58},
            4: {'day': 88, 'night': 70},
            5: {'day': 95, 'night': 78},
            # Monsoon/Summer (Jun, Jul, Aug)
            6: {'day': 100, 'night': 82},
            7: {'day': 95, 'night': 80},
            8: {'day': 92, 'night': 78},
            # Post-monsoon (Sep, Oct, Nov)
            9: {'day': 88, 'night': 75},
            10: {'day': 82, 'night': 65},
            11: {'day': 75, 'night': 55}
        }

        base_temps = seasonal_temps.get(month, {'day': 80, 'night': 65})

        # Calculate temperature based on time of day
        if 6 <= hour <= 18:  # Day time
            temperature = base_temps['day'] + random.uniform(-5, 5)
        else:  # Night time
            temperature = base_temps['night'] + random.uniform(-3, 3)

        # Determine conditions based on season
        conditions = self._get_seasonal_conditions(month)

        # Calculate humidity (higher during monsoon)
        if month in [6, 7, 8, 9]:  # Monsoon season
            humidity = random.uniform(70, 90)
        elif month in [12, 1, 2]:  # Winter (dry)
            humidity = random.uniform(30, 50)
        else:
            humidity = random.uniform(40, 70)

        # Solar radiation (varies by season and time)
        if 6 <= hour <= 18 and conditions in ['clear', 'partly_cloudy']:
            max_radiation = 1000 if conditions == 'clear' else 600
            solar_radiation = max_radiation * math.sin(
                (hour - 6) * math.pi / 12)
        else:
            solar_radiation = 0

        self.current_weather = {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'conditions': conditions,
            'wind_speed': random.uniform(2, 15),
            'wind_direction': random.choice(
                ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'pressure': random.uniform(29.8, 30.2),  # inches Hg
            'visibility': random.uniform(5, 15),  # miles
            'uv_index': max(0,
                            random.uniform(0, 11)) if 6 <= hour <= 18 else 0,
            'solar_radiation': round(solar_radiation, 1),
            'dew_point': temperature - random.uniform(10, 30),
            'feels_like': temperature + random.uniform(-5, 10),
            'timestamp': datetime.utcnow().isoformat(),
            'location': self.location
        }

        # Generate forecast
        self._generate_forecast()

        self.last_update = datetime.utcnow()

    def _get_seasonal_conditions(self, month: int) -> str:
        """Get weather conditions based on season"""
        # Weather patterns for North India
        monsoon_months = [6, 7, 8, 9]  # June to September
        winter_months = [12, 1, 2]

        if month in monsoon_months:
            return random.choices(
                ['rainy', 'cloudy', 'partly_cloudy', 'thunderstorm'],
                weights=[40, 30, 20, 10]
            )[0]
        elif month in winter_months:
            return random.choices(
                ['clear', 'partly_cloudy', 'foggy', 'cloudy'],
                weights=[40, 30, 20, 10]
            )[0]
        else:  # Spring/Summer
            return random.choices(
                ['clear', 'partly_cloudy', 'hot', 'dusty'],
                weights=[50, 30, 15, 5]
            )[0]

    def _generate_forecast(self):
        """Generate 7-day weather forecast"""
        self.forecast_data = []
        current_temp = self.current_weather['temperature']
        current_conditions = self.current_weather['conditions']

        for day in range(7):
            forecast_date = datetime.utcnow() + timedelta(days=day)

            # Temperature trend (gradual changes)
            temp_variation = random.uniform(-8, 8)
            high_temp = current_temp + temp_variation + random.uniform(5, 15)
            low_temp = high_temp - random.uniform(10, 25)

            # Conditions (some persistence with change)
            if random.random() < 0.7:  # 70% chance to keep similar conditions
                conditions = current_conditions
            else:
                conditions = self._get_seasonal_conditions(forecast_date.month)

            precipitation_chance = {
                'clear': 5, 'partly_cloudy': 15, 'cloudy': 35,
                'rainy': 80, 'thunderstorm': 90, 'foggy': 10,
                'hot': 5, 'dusty': 0
            }.get(conditions, 20)

            forecast_day = {
                'date': forecast_date.date().isoformat(),
                'high_temperature': round(high_temp, 1),
                'low_temperature': round(low_temp, 1),
                'conditions': conditions,
                'precipitation_chance': precipitation_chance,
                'humidity': random.uniform(30, 80),
                'wind_speed': random.uniform(5, 20),
                'uv_index': random.uniform(3,
                                           10) if conditions != 'rainy' else random.uniform(
                    1, 4)
            }

            self.forecast_data.append(forecast_day)

        logger.debug(f"Generated {len(self.forecast_data)} day forecast")

    async def get_current_weather(self) -> Dict[str, Any]:
        """Get current weather conditions"""
        try:
            # Check if update is needed
            if (not self.last_update or
                    (
                            datetime.utcnow() - self.last_update).seconds > self.update_interval):
                await self._update_weather_data()

            return self.current_weather.copy()

        except Exception as e:
            logger.error(f"Error getting current weather: {e}")
            return {}

    async def get_weather_forecast(self, days: int = 7) -> List[
        Dict[str, Any]]:
        """Get weather forecast for specified number of days"""
        try:
            # Ensure we have recent forecast data
            if (not self.last_update or
                    (
                            datetime.utcnow() - self.last_update).seconds > self.update_interval):
                await self._update_weather_data()

            return self.forecast_data[:days]

        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return []

    async def get_hourly_forecast(self, hours: int = 24) -> List[
        Dict[str, Any]]:
        """Get hourly weather forecast"""
        try:
            hourly_data = []
            current_time = datetime.utcnow()
            base_temp = self.current_weather.get('temperature', 75)
            base_conditions = self.current_weather.get('conditions', 'clear')

            for hour in range(hours):
                forecast_time = current_time + timedelta(hours=hour)
                hour_of_day = forecast_time.hour

                # Temperature variation throughout the day
                # Peak at 2-4 PM, lowest at 4-6 AM
                temp_factor = 0.5 * (
                            1 + math.cos((hour_of_day - 15) * math.pi / 12))
                daily_range = 20  # 20°F daily temperature range

                temperature = base_temp + (
                            temp_factor - 0.5) * daily_range + random.uniform(
                    -3, 3)

                # Conditions (some variation but mostly stable)
                if random.random() < 0.9:  # 90% chance to keep base conditions
                    conditions = base_conditions
                else:
                    conditions = self._get_seasonal_conditions(
                        forecast_time.month)

                # Solar radiation calculation
                if 6 <= hour_of_day <= 18 and conditions in ['clear',
                                                             'partly_cloudy']:
                    max_radiation = 1000 if conditions == 'clear' else 600
                    solar_radiation = max_radiation * math.sin(
                        (hour_of_day - 6) * math.pi / 12)
                else:
                    solar_radiation = 0

                hourly_data.append({
                    'datetime': forecast_time.isoformat(),
                    'temperature': round(temperature, 1),
                    'conditions': conditions,
                    'humidity': random.uniform(30, 80),
                    'wind_speed': random.uniform(2, 15),
                    'precipitation_chance': random.uniform(0,
                                                           30) if conditions != 'rainy' else random.uniform(
                        60, 90),
                    'solar_radiation': round(solar_radiation, 1),
                    'cloud_cover': self._get_cloud_cover(conditions)
                })

            return hourly_data

        except Exception as e:
            logger.error(f"Error getting hourly forecast: {e}")
            return []

    def _get_cloud_cover(self, conditions: str) -> int:
        """Get cloud cover percentage based on conditions"""
        cloud_cover_map = {
            'clear': 10,
            'partly_cloudy': 40,
            'cloudy': 80,
            'overcast': 100,
            'rainy': 90,
            'thunderstorm': 95,
            'foggy': 100,
            'hot': 15,
            'dusty': 60
        }

        base_cover = cloud_cover_map.get(conditions, 50)
        return max(0, min(100, base_cover + random.randint(-10, 10)))

    async def get_weather_alerts(self) -> List[Dict[str, Any]]:
        """Get weather alerts and warnings"""
        try:
            alerts = []
            current_temp = self.current_weather.get('temperature', 75)
            conditions = self.current_weather.get('conditions', 'clear')
            wind_speed = self.current_weather.get('wind_speed', 5)

            # Heat warning
            if current_temp > 105:  # Above 105°F
                alerts.append({
                    'type': 'heat_warning',
                    'severity': 'high',
                    'title': 'Extreme Heat Warning',
                    'description': f'Temperature {current_temp}°F. Take precautions to stay cool.',
                    'recommendations': [
                        'Increase AC usage for comfort and safety',
                        'Avoid outdoor activities during peak hours',
                        'Stay hydrated'
                    ]
                })
            elif current_temp > 95:
                alerts.append({
                    'type': 'heat_advisory',
                    'severity': 'medium',
                    'title': 'Heat Advisory',
                    'description': f'High temperature {current_temp}°F expected.',
                    'recommendations': [
                        'Pre-cool home before peak hours',
                        'Consider energy-efficient cooling settings'
                    ]
                })

            # Cold warning
            if current_temp < 35:
                alerts.append({
                    'type': 'cold_warning',
                    'severity': 'medium',
                    'title': 'Cold Weather Advisory',
                    'description': f'Temperature {current_temp}°F. Protect against freezing.',
                    'recommendations': [
                        'Ensure adequate heating',
                        'Protect pipes from freezing',
                        'Check HVAC system operation'
                    ]
                })

            # Storm warning
            if conditions == 'thunderstorm':
                alerts.append({
                    'type': 'storm_warning',
                    'severity': 'high',
                    'title': 'Thunderstorm Warning',
                    'description': 'Severe thunderstorms possible. Power outages may occur.',
                    'recommendations': [
                        'Charge backup batteries',
                        'Secure outdoor equipment',
                        'Monitor weather conditions'
                    ]
                })

            # High wind warning
            if wind_speed > 25:
                alerts.append({
                    'type': 'wind_warning',
                    'severity': 'medium',
                    'title': 'High Wind Advisory',
                    'description': f'Sustained winds {wind_speed} mph expected.',
                    'recommendations': [
                        'Secure loose outdoor items',
                        'Monitor solar panel performance',
                        'Be prepared for power outages'
                    ]
                })

            return alerts

        except Exception as e:
            logger.error(f"Error getting weather alerts: {e}")
            return []

    async def get_solar_forecast(self, hours: int = 24) -> List[
        Dict[str, Any]]:
        """Get solar irradiance forecast for solar panel optimization"""
        try:
            solar_forecast = []
            current_time = datetime.utcnow()

            for hour in range(hours):
                forecast_time = current_time + timedelta(hours=hour)
                hour_of_day = forecast_time.hour

                # Base solar irradiance calculation
                if 6 <= hour_of_day <= 18:  # Daylight hours
                    # Peak solar at noon (12 PM)
                    sun_angle = math.sin((hour_of_day - 6) * math.pi / 12)
                    max_irradiance = 1000  # W/m² clear sky

                    # Get weather conditions for this hour
                    hourly_weather = await self._get_hourly_weather_condition(
                        forecast_time)
                    conditions = hourly_weather.get('conditions', 'clear')

                    # Adjust for weather conditions
                    weather_factors = {
                        'clear': 1.0,
                        'partly_cloudy': 0.7,
                        'cloudy': 0.3,
                        'overcast': 0.1,
                        'rainy': 0.1,
                        'thunderstorm': 0.05,
                        'foggy': 0.2,
                        'dusty': 0.6
                    }

                    weather_factor = weather_factors.get(conditions, 0.7)
                    irradiance = max_irradiance * sun_angle * weather_factor

                    # Add some realistic variation
                    irradiance *= random.uniform(0.9, 1.1)
                else:
                    irradiance = 0

                solar_forecast.append({
                    'datetime': forecast_time.isoformat(),
                    'solar_irradiance': round(max(0, irradiance), 1),
                    'cloud_cover': self._get_cloud_cover(
                        hourly_weather.get('conditions', 'clear')),
                    'temperature': hourly_weather.get('temperature', 75),
                    'conditions': hourly_weather.get('conditions', 'clear'),
                    'optimal_generation': irradiance > 200
                    # Good for solar generation
                })

            return solar_forecast

        except Exception as e:
            logger.error(f"Error getting solar forecast: {e}")
            return []

    async def _get_hourly_weather_condition(self, forecast_time: datetime) -> \
    Dict[str, Any]:
        """Get weather conditions for a specific hour"""
        # Simplified - in real implementation, this would use detailed hourly forecast
        base_temp = self.current_weather.get('temperature', 75)
        hour_of_day = forecast_time.hour

        # Temperature variation throughout the day
        temp_factor = 0.5 * (1 + math.cos((hour_of_day - 15) * math.pi / 12))
        daily_range = 20
        temperature = base_temp + (temp_factor - 0.5) * daily_range

        return {
            'temperature': round(temperature, 1),
            'conditions': self.current_weather.get('conditions', 'clear'),
            'humidity': random.uniform(30, 80)
        }

    async def _update_weather_data(self):
        """Update weather data (simulate API call)"""
        try:
            # In a real implementation, this would call a weather API
            # For now, we'll simulate gradual weather changes

            current_temp = self.current_weather.get('temperature', 75)
            current_conditions = self.current_weather.get('conditions',
                                                          'clear')

            # Gradual temperature change
            temp_change = random.uniform(-3, 3)
            new_temp = current_temp + temp_change

            # Occasional condition changes
            if random.random() < 0.1:  # 10% chance of condition change
                new_conditions = self._get_seasonal_conditions(
                    datetime.utcnow().month)
            else:
                new_conditions = current_conditions

            # Update current weather
            self.current_weather.update({
                'temperature': round(new_temp, 1),
                'conditions': new_conditions,
                'humidity': max(20, min(100,
                                        self.current_weather.get('humidity',
                                                                 50) + random.uniform(
                                            -5, 5))),
                'wind_speed': max(0, self.current_weather.get('wind_speed',
                                                              10) + random.uniform(
                    -3, 3)),
                'timestamp': datetime.utcnow().isoformat()
            })

            # Update solar radiation
            hour = datetime.utcnow().hour
            if 6 <= hour <= 18 and new_conditions in ['clear',
                                                      'partly_cloudy']:
                max_radiation = 1000 if new_conditions == 'clear' else 600
                solar_radiation = max_radiation * math.sin(
                    (hour - 6) * math.pi / 12)
            else:
                solar_radiation = 0

            self.current_weather['solar_radiation'] = round(solar_radiation, 1)

            # Regenerate forecast occasionally
            if random.random() < 0.2:  # 20% chance to update forecast
                self._generate_forecast()

            self.last_update = datetime.utcnow()
            logger.debug("Weather data updated")

        except Exception as e:
            logger.error(f"Error updating weather data: {e}")

    async def get_weather_impact_analysis(self) -> Dict[str, Any]:
        """Analyze weather impact on energy consumption"""
        try:
            current_weather = await self.get_current_weather()
            forecast = await self.get_weather_forecast(3)  # 3-day forecast

            temperature = current_weather.get('temperature', 75)
            conditions = current_weather.get('conditions', 'clear')
            humidity = current_weather.get('humidity', 50)

            # Analyze heating/cooling needs
            comfort_range = (68, 78)  # Comfortable temperature range

            if temperature < comfort_range[0]:
                heating_need = 'high' if temperature < 60 else 'medium'
                cooling_need = 'none'
            elif temperature > comfort_range[1]:
                cooling_need = 'high' if temperature > 85 else 'medium'
                heating_need = 'none'
            else:
                heating_need = 'low'
                cooling_need = 'low'

            # Solar generation impact
            solar_conditions = ['clear', 'partly_cloudy']
            solar_impact = 'high' if conditions in solar_conditions else 'low'

            # Overall energy impact
            if heating_need == 'high' or cooling_need == 'high':
                energy_impact = 'high'
            elif heating_need == 'medium' or cooling_need == 'medium':
                energy_impact = 'medium'
            else:
                energy_impact = 'low'

            analysis = {
                'current_conditions': current_weather,
                'energy_impact': energy_impact,
                'heating_need': heating_need,
                'cooling_need': cooling_need,
                'solar_generation_potential': solar_impact,
                'humidity_impact': 'high' if humidity > 70 else 'low',
                'recommendations': [],
                'forecast_trends': self._analyze_forecast_trends(forecast)
            }

            # Generate recommendations
            if cooling_need == 'high':
                analysis['recommendations'].extend([
                    'Pre-cool home during off-peak hours',
                    'Use higher temperature setpoints during peak pricing',
                    'Ensure solar panels are clean for maximum generation'
                ])

            if heating_need == 'high':
                analysis['recommendations'].extend([
                    'Optimize heating schedule for energy efficiency',
                    'Consider lower temperature setpoints when unoccupied',
                    'Use solar generation for heating when available'
                ])

            if solar_impact == 'high':
                analysis['recommendations'].append(
                    'Schedule energy-intensive tasks during high solar generation periods'
                )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}")
            return {}

    def _analyze_forecast_trends(self, forecast: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """Analyze trends in weather forecast"""
        if not forecast:
            return {}

        temperatures = [(day['high_temperature'] + day['low_temperature']) / 2
                        for day in forecast]

        # Calculate trend
        if len(temperatures) >= 2:
            temp_trend = 'warming' if temperatures[-1] > temperatures[
                0] else 'cooling'
            temp_change = temperatures[-1] - temperatures[0]
        else:
            temp_trend = 'stable'
            temp_change = 0

        # Analyze conditions
        conditions = [day['conditions'] for day in forecast]
        rainy_days = sum(
            1 for cond in conditions if 'rain' in cond or 'storm' in cond)

        return {
            'temperature_trend': temp_trend,
            'temperature_change': round(temp_change, 1),
            'rainy_days_forecast': rainy_days,
            'average_temp': round(sum(temperatures) / len(temperatures), 1),
            'max_temp': max(day['high_temperature'] for day in forecast),
            'min_temp': min(day['low_temperature'] for day in forecast)
        }

    async def health_check(self) -> bool:
        """Check if weather service is healthy"""
        try:
            current_weather = await self.get_current_weather()
            return bool(current_weather and 'temperature' in current_weather)
        except Exception as e:
            logger.error(f"Weather service health check failed: {e}")
            return False