import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
import re
import math


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix"""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def generate_short_id(length: int = 8) -> str:
    """Generate a short unique ID"""
    return str(uuid.uuid4()).replace('-', '')[:length]


def hash_string(text: str) -> str:
    """Generate SHA-256 hash of a string"""
    return hashlib.sha256(text.encode()).hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    if isinstance(value, (int, float)):
        return value != 0
    return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.0
    return clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)


def denormalize(normalized_value: float, min_val: float,
                max_val: float) -> float:
    """Convert normalized value back to original range"""
    return min_val + normalized_value * (max_val - min_val)


def round_to_nearest(value: float, nearest: float) -> float:
    """Round value to nearest increment"""
    return round(value / nearest) * nearest


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / old_value) * 100


def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average of values"""
    if len(values) < window_size:
        return values

    result = []
    for i in range(len(values) - window_size + 1):
        avg = sum(values[i:i + window_size]) / window_size
        result.append(avg)

    return result


def exponential_moving_average(values: List[float], alpha: float = 0.3) -> \
List[float]:
    """Calculate exponential moving average"""
    if not values:
        return []

    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)

    return result


def calculate_energy_cost(consumption_kwh: float, price_per_kwh: float,
                          taxes: float = 0.08, fees: float = 0.02) -> Dict[
    str, float]:
    """Calculate total energy cost including taxes and fees"""
    base_cost = consumption_kwh * price_per_kwh
    tax_amount = base_cost * taxes
    fee_amount = base_cost * fees
    total_cost = base_cost + tax_amount + fee_amount

    return {
        'base_cost': round(base_cost, 2),
        'taxes': round(tax_amount, 2),
        'fees': round(fee_amount, 2),
        'total_cost': round(total_cost, 2),
        'effective_rate': round(total_cost / consumption_kwh,
                                4) if consumption_kwh > 0 else 0
    }


def watts_to_kwh(watts: float, hours: float) -> float:
    """Convert watts to kilowatt-hours"""
    return (watts * hours) / 1000


def kwh_to_watts(kwh: float, hours: float) -> float:
    """Convert kilowatt-hours to watts"""
    return (kwh * 1000) / hours if hours > 0 else 0


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius"""
    return (fahrenheit - 32) * 5 / 9


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9 / 5) + 32


def calculate_comfort_score(current_temp: float, target_temp: float,
                            tolerance: float = 2.0) -> float:
    """Calculate comfort score based on temperature deviation"""
    deviation = abs(current_temp - target_temp)
    if deviation <= tolerance:
        return 1.0 - (
                    deviation / tolerance) * 0.3  # Linear decrease within tolerance
    else:
        return max(0.0, 0.7 - ((
                                           deviation - tolerance) / tolerance) * 0.7)  # Steeper decrease outside tolerance


def calculate_efficiency_score(actual_consumption: float,
                               expected_consumption: float) -> float:
    """Calculate efficiency score comparing actual vs expected consumption"""
    if expected_consumption == 0:
        return 0.0

    ratio = actual_consumption / expected_consumption

    if ratio <= 1.0:
        return 100.0  # Perfect or better than expected
    elif ratio <= 1.2:
        return 100.0 - (
                    ratio - 1.0) * 250  # Linear decrease up to 20% overconsumption
    else:
        return max(0.0,
                   50.0 - (ratio - 1.2) * 62.5)  # Steeper decrease beyond 20%


def parse_time_string(time_str: str) -> Optional[datetime]:
    """Parse time string in various formats"""
    formats = [
        '%H:%M',
        '%H:%M:%S',
        '%I:%M %p',
        '%I:%M:%S %p',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


def format_energy(kwh: float) -> str:
    """Format energy in human-readable format"""
    if kwh < 1:
        return f"{kwh * 1000:.0f} Wh"
    elif kwh < 1000:
        return f"{kwh:.2f} kWh"
    else:
        return f"{kwh / 1000:.2f} MWh"


def format_power(watts: float) -> str:
    """Format power in human-readable format"""
    if watts < 1000:
        return f"{watts:.0f} W"
    elif watts < 1000000:
        return f"{watts / 1000:.2f} kW"
    else:
        return f"{watts / 1000000:.2f} MW"


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "$")
    return f"{symbol}{amount:.2f}"


def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    # Check if it's a valid length (10-15 digits)
    return 10 <= len(digits) <= 15


def sanitize_device_id(device_id: str) -> str:
    """Sanitize device ID to be safe for use as identifier"""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', device_id)
    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f"device_{sanitized}"
    return sanitized or "device_unknown"


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[
    str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(
                value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def filter_dict_by_keys(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[
    str, Any]:
    """Filter dictionary to only include allowed keys"""
    return {key: value for key, value in data.items() if key in allowed_keys}


def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""

    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}

        return dict(items)

    return _flatten(data)


def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[
    str, Any]:
    """Unflatten dictionary with separator keys"""
    result = {}

    for key, value in data.items():
        parts = key.split(separator)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0,
                       exceptions: tuple = (Exception,)):
    """Decorator for retrying functions with exponential backoff"""

    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise e

                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)

        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise e

                    wait_time = backoff_factor ** attempt
                    import time
                    time.sleep(wait_time)

        return async_wrapper if asyncio.iscoroutinefunction(
            func) else sync_wrapper

    return decorator


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches of specified size"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def calculate_solar_position(latitude: float, longitude: float,
                             timestamp: datetime) -> Dict[str, float]:
    """Calculate solar position (simplified calculation)"""
    # Convert to radians
    lat_rad = math.radians(latitude)

    # Day of year
    day_of_year = timestamp.timetuple().tm_yday

    # Solar declination
    declination = math.radians(23.45) * math.sin(
        math.radians(360 * (284 + day_of_year) / 365))

    # Hour angle
    hour = timestamp.hour + timestamp.minute / 60.0
    hour_angle = math.radians(15 * (hour - 12))

    # Solar elevation
    elevation = math.asin(
        math.sin(lat_rad) * math.sin(declination) +
        math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )

    # Solar azimuth
    azimuth = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat_rad) - math.tan(
            declination) * math.cos(lat_rad)
    )

    return {
        'elevation_deg': math.degrees(elevation),
        'azimuth_deg': math.degrees(azimuth),
        'is_daylight': elevation > 0
    }


def calculate_hvac_efficiency(outdoor_temp: float, indoor_temp: float,
                              mode: str = 'cooling') -> float:
    """Calculate HVAC efficiency based on temperature difference"""
    temp_diff = abs(outdoor_temp - indoor_temp)

    if mode == 'cooling':
        # Efficiency decreases as outdoor temperature increases
        base_efficiency = 3.0  # COP for cooling
        efficiency_factor = max(0.5, 1.0 - (temp_diff - 20) / 100)
    else:  # heating
        # Efficiency decreases as outdoor temperature decreases
        base_efficiency = 2.5  # COP for heating
        efficiency_factor = max(0.5, 1.0 - (temp_diff - 15) / 80)

    return base_efficiency * efficiency_factor


def detect_anomaly(values: List[float], threshold: float = 2.0) -> List[bool]:
    """Detect anomalies using z-score method"""
    if len(values) < 3:
        return [False] * len(values)

    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return [False] * len(values)

    anomalies = []
    for value in values:
        z_score = abs((value - mean_val) / std_dev)
        anomalies.append(z_score > threshold)

    return anomalies


def smooth_values(values: List[float], smoothing_factor: float = 0.3) -> List[
    float]:
    """Smooth values using exponential smoothing"""
    if not values:
        return []

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_value = smoothing_factor * values[i] + (
                    1 - smoothing_factor) * smoothed[-1]
        smoothed.append(smoothed_value)

    return smoothed


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def is_allowed(self) -> bool:
        """Check if call is allowed under rate limit"""
        now = datetime.utcnow().timestamp()

        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls
                      if now - call_time < self.time_window]

        # Check if under limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True

        return False

    def time_until_next_call(self) -> float:
        """Get time in seconds until next call is allowed"""
        if len(self.calls) < self.max_calls:
            return 0.0

        oldest_call = min(self.calls)
        return max(0.0, self.time_window - (
                    datetime.utcnow().timestamp() - oldest_call))


class CircularBuffer:
    """Circular buffer for storing fixed number of recent values"""

    def __init__(self, size: int):
        self.size = size
        self.buffer = []
        self.index = 0
        self.is_full = False

    def append(self, value: Any):
        """Add value to buffer"""
        if not self.is_full:
            self.buffer.append(value)
            if len(self.buffer) == self.size:
                self.is_full = True
        else:
            self.buffer[self.index] = value
            self.index = (self.index + 1) % self.size

    def get_values(self) -> List[Any]:
        """Get all values in chronological order"""
        if not self.is_full:
            return self.buffer.copy()

        return self.buffer[self.index:] + self.buffer[:self.index]

    def get_latest(self, n: int = 1) -> List[Any]:
        """Get n latest values"""
        values = self.get_values()
        return values[-n:] if values else []

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.index = 0
        self.is_full = False