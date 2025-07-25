from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class DeviceType(Enum):
    HVAC = "hvac"
    APPLIANCE = "appliance"
    LIGHTING = "lighting"
    EV_CHARGING = "ev_charging"
    SOLAR_BATTERY = "solar_battery"
    SECURITY = "security"
    ENTERTAINMENT = "entertainment"


class EnergyEventType(Enum):
    CONSUMPTION = "consumption"
    GENERATION = "generation"
    STORAGE = "storage"
    EXPORT = "export"
    IMPORT = "import"


class PriceTier(Enum):
    OFF_PEAK = "off_peak"
    STANDARD = "standard"
    PEAK = "peak"


@dataclass
class EnergyConsumption:
    """Energy consumption data point"""
    timestamp: datetime
    device_id: str
    device_type: DeviceType
    power_consumption: float  # Watts
    energy_consumption: float  # kWh
    room: Optional[str] = None
    device_state: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[float] = None
    efficiency_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'device_type': self.device_type.value,
            'power_consumption': self.power_consumption,
            'energy_consumption': self.energy_consumption,
            'room': self.room,
            'device_state': self.device_state,
            'cost_estimate': self.cost_estimate,
            'efficiency_score': self.efficiency_score
        }


@dataclass
class EnergyGeneration:
    """Energy generation data point (solar, etc.)"""
    timestamp: datetime
    device_id: str
    power_generation: float  # Watts
    energy_generated: float  # kWh
    efficiency: Optional[float] = None
    weather_conditions: Optional[Dict[str, Any]] = None
    irradiance: Optional[float] = None  # W/m²
    panel_temperature: Optional[float] = None  # °F

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'power_generation': self.power_generation,
            'energy_generated': self.energy_generated,
            'efficiency': self.efficiency,
            'weather_conditions': self.weather_conditions,
            'irradiance': self.irradiance,
            'panel_temperature': self.panel_temperature
        }


@dataclass
class EnergyStorage:
    """Energy storage data point (battery)"""
    timestamp: datetime
    device_id: str
    state_of_charge: float  # Percentage
    power_flow: float  # Watts (positive = discharging, negative = charging)
    energy_capacity: float  # kWh
    available_capacity: float  # kWh
    efficiency: Optional[float] = None
    cycle_count: Optional[int] = None
    temperature: Optional[float] = None  # °F

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'state_of_charge': self.state_of_charge,
            'power_flow': self.power_flow,
            'energy_capacity': self.energy_capacity,
            'available_capacity': self.available_capacity,
            'efficiency': self.efficiency,
            'cycle_count': self.cycle_count,
            'temperature': self.temperature
        }


@dataclass
class EnergyPrice:
    """Energy pricing data point"""
    timestamp: datetime
    price_per_kwh: float
    price_tier: PriceTier
    utility_company: str
    demand_charge: Optional[float] = None
    transmission_charge: Optional[float] = None
    distribution_charge: Optional[float] = None
    taxes_and_fees: Optional[float] = None
    forecast: bool = False
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'price_per_kwh': self.price_per_kwh,
            'price_tier': self.price_tier.value,
            'utility_company': self.utility_company,
            'demand_charge': self.demand_charge,
            'transmission_charge': self.transmission_charge,
            'distribution_charge': self.distribution_charge,
            'taxes_and_fees': self.taxes_and_fees,
            'forecast': self.forecast,
            'confidence': self.confidence
        }


@dataclass
class EnergyEvent:
    """General energy event"""
    event_id: str
    timestamp: datetime
    event_type: EnergyEventType
    device_id: Optional[str] = None
    agent_id: Optional[str] = None
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'device_id': self.device_id,
            'agent_id': self.agent_id,
            'description': self.description,
            'data': self.data,
            'severity': self.severity,
            'acknowledged': self.acknowledged
        }


@dataclass
class EnergyForecast:
    """Energy forecast data"""
    timestamp: datetime
    forecast_type: str  # consumption, generation, price
    forecast_horizon: int  # Hours
    forecasted_values: List[float]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    model_used: Optional[str] = None
    accuracy_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'forecast_type': self.forecast_type,
            'forecast_horizon': self.forecast_horizon,
            'forecasted_values': self.forecasted_values,
            'confidence_intervals': self.confidence_intervals,
            'model_used': self.model_used,
            'accuracy_score': self.accuracy_score
        }


@dataclass
class EnergyOptimization:
    """Energy optimization recommendation"""
    optimization_id: str
    timestamp: datetime
    agent_id: str
    optimization_type: str  # schedule, control, configuration
    target_devices: List[str]
    recommendations: List[Dict[str, Any]]
    potential_savings: Optional[float] = None  # Percentage
    comfort_impact: Optional[str] = None  # minimal, moderate, significant
    implementation_status: str = "pending"  # pending, implemented, rejected

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_id': self.optimization_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'optimization_type': self.optimization_type,
            'target_devices': self.target_devices,
            'recommendations': self.recommendations,
            'potential_savings': self.potential_savings,
            'comfort_impact': self.comfort_impact,
            'implementation_status': self.implementation_status
        }


@dataclass
class GridInteraction:
    """Grid interaction data"""
    timestamp: datetime
    net_power_flow: float  # kW (positive = export, negative = import)
    solar_generation: float  # kW
    battery_discharge: float  # kW
    home_consumption: float  # kW
    grid_frequency: Optional[float] = None  # Hz
    voltage: Optional[float] = None  # V
    power_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'net_power_flow': self.net_power_flow,
            'solar_generation': self.solar_generation,
            'battery_discharge': self.battery_discharge,
            'home_consumption': self.home_consumption,
            'grid_frequency': self.grid_frequency,
            'voltage': self.voltage,
            'power_factor': self.power_factor
        }


@dataclass
class DemandResponseEvent:
    """Demand response event data"""
    event_id: str
    timestamp: datetime
    start_time: datetime
    end_time: datetime
    event_type: str  # peak_reduction, emergency, voluntary
    utility_company: str
    target_reduction: float  # Percentage
    incentive: float  # $/kWh
    mandatory: bool = False
    participation_status: str = "pending"  # pending, participating, declined
    actual_reduction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'event_type': self.event_type,
            'utility_company': self.utility_company,
            'target_reduction': self.target_reduction,
            'incentive': self.incentive,
            'mandatory': self.mandatory,
            'participation_status': self.participation_status,
            'actual_reduction': self.actual_reduction
        }


@dataclass
class EnergyBudget:
    """Energy budget for planning and tracking"""
    budget_id: str
    period_start: datetime
    period_end: datetime
    budget_type: str  # daily, weekly, monthly
    target_consumption: float  # kWh
    target_cost: float  # $
    actual_consumption: float = 0.0  # kWh
    actual_cost: float = 0.0  # $
    efficiency_target: Optional[float] = None  # kWh/day per sq ft
    renewable_target: Optional[float] = None  # Percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            'budget_id': self.budget_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'budget_type': self.budget_type,
            'target_consumption': self.target_consumption,
            'target_cost': self.target_cost,
            'actual_consumption': self.actual_consumption,
            'actual_cost': self.actual_cost,
            'efficiency_target': self.efficiency_target,
            'renewable_target': self.renewable_target
        }

    @property
    def consumption_variance(self) -> float:
        """Calculate consumption variance from target"""
        if self.target_consumption > 0:
            return ((
                                self.actual_consumption - self.target_consumption) / self.target_consumption) * 100
        return 0.0

    @property
    def cost_variance(self) -> float:
        """Calculate cost variance from target"""
        if self.target_cost > 0:
            return ((
                                self.actual_cost - self.target_cost) / self.target_cost) * 100
        return 0.0

    @property
    def is_on_track(self) -> bool:
        """Check if budget is on track"""
        return abs(self.consumption_variance) <= 10 and abs(
            self.cost_variance) <= 10


# Utility functions for data models

def create_consumption_from_dict(data: Dict[str, Any]) -> EnergyConsumption:
    """Create EnergyConsumption from dictionary"""
    return EnergyConsumption(
        timestamp=datetime.fromisoformat(data['timestamp']),
        device_id=data['device_id'],
        device_type=DeviceType(data['device_type']),
        power_consumption=data['power_consumption'],
        energy_consumption=data['energy_consumption'],
        room=data.get('room'),
        device_state=data.get('device_state'),
        cost_estimate=data.get('cost_estimate'),
        efficiency_score=data.get('efficiency_score')
    )


def create_generation_from_dict(data: Dict[str, Any]) -> EnergyGeneration:
    """Create EnergyGeneration from dictionary"""
    return EnergyGeneration(
        timestamp=datetime.fromisoformat(data['timestamp']),
        device_id=data['device_id'],
        power_generation=data['power_generation'],
        energy_generated=data['energy_generated'],
        efficiency=data.get('efficiency'),
        weather_conditions=data.get('weather_conditions'),
        irradiance=data.get('irradiance'),
        panel_temperature=data.get('panel_temperature')
    )


def create_storage_from_dict(data: Dict[str, Any]) -> EnergyStorage:
    """Create EnergyStorage from dictionary"""
    return EnergyStorage(
        timestamp=datetime.fromisoformat(data['timestamp']),
        device_id=data['device_id'],
        state_of_charge=data['state_of_charge'],
        power_flow=data['power_flow'],
        energy_capacity=data['energy_capacity'],
        available_capacity=data['available_capacity'],
        efficiency=data.get('efficiency'),
        cycle_count=data.get('cycle_count'),
        temperature=data.get('temperature')
    )


def create_price_from_dict(data: Dict[str, Any]) -> EnergyPrice:
    """Create EnergyPrice from dictionary"""
    return EnergyPrice(
        timestamp=datetime.fromisoformat(data['timestamp']),
        price_per_kwh=data['price_per_kwh'],
        price_tier=PriceTier(data['price_tier']),
        utility_company=data['utility_company'],
        demand_charge=data.get('demand_charge'),
        transmission_charge=data.get('transmission_charge'),
        distribution_charge=data.get('distribution_charge'),
        taxes_and_fees=data.get('taxes_and_fees'),
        forecast=data.get('forecast', False),
        confidence=data.get('confidence')
    )


def aggregate_consumption(consumption_data: List[EnergyConsumption],
                          interval: str = 'hourly') -> List[Dict[str, Any]]:
    """Aggregate consumption data by time interval"""
    from collections import defaultdict

    aggregated = defaultdict(
        lambda: {'total_consumption': 0, 'device_count': 0, 'cost': 0})

    for data_point in consumption_data:
        if interval == 'hourly':
            key = data_point.timestamp.replace(minute=0, second=0,
                                               microsecond=0)
        elif interval == 'daily':
            key = data_point.timestamp.replace(hour=0, minute=0, second=0,
                                               microsecond=0)
        elif interval == 'monthly':
            key = data_point.timestamp.replace(day=1, hour=0, minute=0,
                                               second=0, microsecond=0)
        else:
            key = data_point.timestamp

        aggregated[key]['total_consumption'] += data_point.energy_consumption
        aggregated[key]['device_count'] += 1
        if data_point.cost_estimate:
            aggregated[key]['cost'] += data_point.cost_estimate

    result = []
    for timestamp, data in sorted(aggregated.items()):
        result.append({
            'timestamp': timestamp.isoformat(),
            'total_consumption': data['total_consumption'],
            'average_consumption': data['total_consumption'] / data[
                'device_count'],
            'device_count': data['device_count'],
            'total_cost': data['cost']
        })

    return result


def calculate_efficiency_metrics(consumption_data: List[EnergyConsumption],
                                 generation_data: List[EnergyGeneration]) -> \
Dict[str, float]:
    """Calculate efficiency metrics from consumption and generation data"""
    total_consumption = sum(d.energy_consumption for d in consumption_data)
    total_generation = sum(d.energy_generated for d in generation_data)

    metrics = {
        'total_consumption_kwh': total_consumption,
        'total_generation_kwh': total_generation,
        'net_consumption_kwh': total_consumption - total_generation,
        'self_consumption_ratio': 0.0,
        'energy_independence_ratio': 0.0
    }

    if total_consumption > 0:
        metrics['self_consumption_ratio'] = min(1.0,
                                                total_generation / total_consumption)
        metrics['energy_independence_ratio'] = max(0.0, (
                    total_generation - total_consumption) / total_consumption)

    return metrics