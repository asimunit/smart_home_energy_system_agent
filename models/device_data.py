from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class DeviceCapability(Enum):
    ON_OFF = "on_off"
    DIMMING = "dimming"
    COLOR_CONTROL = "color_control"
    TEMPERATURE_CONTROL = "temperature_control"
    SCHEDULING = "scheduling"
    REMOTE_CONTROL = "remote_control"
    ENERGY_MONITORING = "energy_monitoring"
    LOAD_BALANCING = "load_balancing"


class PowerState(Enum):
    ON = "on"
    OFF = "off"
    STANDBY = "standby"
    SLEEP = "sleep"


@dataclass
class DeviceInfo:
    """Basic device information"""
    device_id: str
    device_type: str
    device_subtype: str
    name: str
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    firmware_version: str = "Unknown"
    room: str = "Unknown"
    installation_date: Optional[datetime] = None
    warranty_expiry: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'device_subtype': self.device_subtype,
            'name': self.name,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'firmware_version': self.firmware_version,
            'room': self.room,
            'installation_date': self.installation_date.isoformat() if self.installation_date else None,
            'warranty_expiry': self.warranty_expiry.isoformat() if self.warranty_expiry else None
        }


@dataclass
class DeviceSpecifications:
    """Device technical specifications"""
    power_rating: float  # Watts
    voltage: float = 120.0  # Volts
    current_rating: float = 15.0  # Amps
    frequency: float = 60.0  # Hz
    efficiency_rating: Optional[str] = None  # Energy Star, SEER, etc.
    capabilities: List[DeviceCapability] = field(default_factory=list)
    operating_temperature_range: Optional[
        Dict[str, float]] = None  # min/max °F
    dimensions: Optional[
        Dict[str, float]] = None  # length/width/height in inches
    weight: Optional[float] = None  # pounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'power_rating': self.power_rating,
            'voltage': self.voltage,
            'current_rating': self.current_rating,
            'frequency': self.frequency,
            'efficiency_rating': self.efficiency_rating,
            'capabilities': [cap.value for cap in self.capabilities],
            'operating_temperature_range': self.operating_temperature_range,
            'dimensions': self.dimensions,
            'weight': self.weight
        }


@dataclass
class DeviceState:
    """Current device state"""
    timestamp: datetime
    device_id: str
    power_state: PowerState
    power_consumption: float  # Current watts
    status: DeviceStatus = DeviceStatus.ONLINE
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    state_data: Dict[str, Any] = field(default_factory=dict)
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'power_state': self.power_state.value,
            'power_consumption': self.power_consumption,
            'status': self.status.value,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'state_data': self.state_data,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


@dataclass
class HVACState:
    """HVAC-specific device state"""
    current_temperature: float  # °F
    target_temperature: float  # °F
    mode: str  # auto, heat, cool, off
    fan_speed: str  # auto, low, medium, high
    humidity: float  # Percentage
    filter_status: str = "good"  # good, replace_soon, replace_now

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_temperature': self.current_temperature,
            'target_temperature': self.target_temperature,
            'mode': self.mode,
            'fan_speed': self.fan_speed,
            'humidity': self.humidity,
            'filter_status': self.filter_status
        }


@dataclass
class LightingState:
    """Lighting-specific device state"""
    brightness: int  # 0-100%
    color_temperature: int  # Kelvin
    color: Optional[Dict[str, int]] = None  # RGB values
    scene: Optional[str] = None
    motion_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'brightness': self.brightness,
            'color_temperature': self.color_temperature,
            'color': self.color,
            'scene': self.scene,
            'motion_detected': self.motion_detected
        }


@dataclass
class ApplianceState:
    """Appliance-specific device state"""
    cycle_state: str  # idle, running, complete, error
    cycle_type: Optional[str] = None  # normal, eco, quick, etc.
    time_remaining: int = 0  # minutes
    door_locked: bool = False
    door_open: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_state': self.cycle_state,
            'cycle_type': self.cycle_type,
            'time_remaining': self.time_remaining,
            'door_locked': self.door_locked,
            'door_open': self.door_open
        }


@dataclass
class EVChargerState:
    """EV Charger-specific device state"""
    charging_state: str  # idle, charging, complete, error
    connector_state: str  # connected, disconnected
    current_power: float  # Current charging power in watts
    session_energy: float  # Energy delivered in current session (kWh)
    vehicle_soc: Optional[float] = None  # Vehicle state of charge if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            'charging_state': self.charging_state,
            'connector_state': self.connector_state,
            'current_power': self.current_power,
            'session_energy': self.session_energy,
            'vehicle_soc': self.vehicle_soc
        }


@dataclass
class SolarPanelState:
    """Solar Panel-specific device state"""
    power_generation: float  # Current generation in watts
    daily_generation: float  # Total generation today (kWh)
    panel_temperature: float  # °F
    irradiance: float  # W/m²
    efficiency: float  # Current efficiency percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            'power_generation': self.power_generation,
            'daily_generation': self.daily_generation,
            'panel_temperature': self.panel_temperature,
            'irradiance': self.irradiance,
            'efficiency': self.efficiency
        }


@dataclass
class BatteryState:
    """Battery-specific device state"""
    state_of_charge: float  # Percentage
    power_flow: float  # Watts (positive = discharging, negative = charging)
    available_capacity: float  # kWh currently available
    temperature: float  # °F
    cycle_count: int = 0
    health: float = 100.0  # Battery health percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_of_charge': self.state_of_charge,
            'power_flow': self.power_flow,
            'available_capacity': self.available_capacity,
            'temperature': self.temperature,
            'cycle_count': self.cycle_count,
            'health': self.health
        }


@dataclass
class DeviceSchedule:
    """Device operation schedule"""
    schedule_id: str
    device_id: str
    schedule_type: str  # daily, weekly, one_time
    enabled: bool = True
    schedule_data: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    last_executed: Optional[datetime] = None
    next_execution: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schedule_id': self.schedule_id,
            'device_id': self.device_id,
            'schedule_type': self.schedule_type,
            'enabled': self.enabled,
            'schedule_data': self.schedule_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'next_execution': self.next_execution.isoformat() if self.next_execution else None
        }


@dataclass
class DeviceGroup:
    """Group of related devices"""
    group_id: str
    name: str
    device_ids: List[str]
    group_type: str  # room, system, category
    control_strategy: str = "synchronized"  # synchronized, load_balanced, sequential
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'group_id': self.group_id,
            'name': self.name,
            'device_ids': self.device_ids,
            'group_type': self.group_type,
            'control_strategy': self.control_strategy,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class DeviceAlert:
    """Device alert or notification"""
    alert_id: str
    device_id: str
    timestamp: datetime
    alert_type: str  # error, warning, maintenance, info
    severity: str  # critical, high, medium, low
    title: str
    description: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'device_id': self.device_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


@dataclass
class DeviceConfiguration:
    """Device configuration settings"""
    device_id: str
    configuration: Dict[str, Any]
    version: int = 1
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None  # Agent or user who applied configuration

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'configuration': self.configuration,
            'version': self.version,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'applied_by': self.applied_by
        }


@dataclass
class DeviceMaintenanceRecord:
    """Device maintenance record"""
    record_id: str
    device_id: str
    maintenance_type: str  # routine, repair, upgrade, replacement
    performed_at: datetime
    performed_by: str
    description: str
    cost: Optional[float] = None
    next_maintenance_due: Optional[datetime] = None
    parts_replaced: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'device_id': self.device_id,
            'maintenance_type': self.maintenance_type,
            'performed_at': self.performed_at.isoformat(),
            'performed_by': self.performed_by,
            'description': self.description,
            'cost': self.cost,
            'next_maintenance_due': self.next_maintenance_due.isoformat() if self.next_maintenance_due else None,
            'parts_replaced': self.parts_replaced
        }


@dataclass
class DeviceUsageStatistics:
    """Device usage statistics"""
    device_id: str
    period_start: datetime
    period_end: datetime
    total_runtime_hours: float
    total_energy_consumed: float  # kWh
    average_power_consumption: float  # Watts
    peak_power_consumption: float  # Watts
    cycle_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_runtime_hours': self.total_runtime_hours,
            'total_energy_consumed': self.total_energy_consumed,
            'average_power_consumption': self.average_power_consumption,
            'peak_power_consumption': self.peak_power_consumption,
            'cycle_count': self.cycle_count,
            'error_count': self.error_count
        }

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on usage patterns"""
        if self.total_runtime_hours == 0:
            return 0.0

        # Simple efficiency metric based on average vs peak consumption
        if self.peak_power_consumption > 0:
            efficiency = (
                                     self.average_power_consumption / self.peak_power_consumption) * 100
            return min(100.0, efficiency)

        return 50.0  # Default moderate score


# Utility functions for device data models

def create_device_state_from_dict(device_type: str,
                                  data: Dict[str, Any]) -> DeviceState:
    """Create appropriate device state from dictionary based on device type"""
    base_state = DeviceState(
        timestamp=datetime.fromisoformat(data['timestamp']),
        device_id=data['device_id'],
        power_state=PowerState(data['power_state']),
        power_consumption=data['power_consumption'],
        status=DeviceStatus(data.get('status', 'online')),
        error_code=data.get('error_code'),
        error_message=data.get('error_message'),
        state_data=data.get('state_data', {}),
        last_update=datetime.fromisoformat(data['last_update']) if data.get(
            'last_update') else None
    )

    # Add device-specific state based on type
    if device_type == 'hvac' and 'hvac_state' in data:
        hvac_data = data['hvac_state']
        base_state.state_data['hvac'] = HVACState(
            current_temperature=hvac_data['current_temperature'],
            target_temperature=hvac_data['target_temperature'],
            mode=hvac_data['mode'],
            fan_speed=hvac_data['fan_speed'],
            humidity=hvac_data['humidity'],
            filter_status=hvac_data.get('filter_status', 'good')
        )

    elif device_type == 'lighting' and 'lighting_state' in data:
        lighting_data = data['lighting_state']
        base_state.state_data['lighting'] = LightingState(
            brightness=lighting_data['brightness'],
            color_temperature=lighting_data['color_temperature'],
            color=lighting_data.get('color'),
            scene=lighting_data.get('scene'),
            motion_detected=lighting_data.get('motion_detected', False)
        )

    # Add more device-specific states as needed

    return base_state


def calculate_device_health_score(device_id: str,
                                  usage_stats: DeviceUsageStatistics,
                                  alerts: List[DeviceAlert]) -> float:
    """Calculate overall device health score"""
    health_factors = []

    # Efficiency factor
    efficiency_score = usage_stats.efficiency_score
    health_factors.append(efficiency_score * 0.3)  # 30% weight

    # Error rate factor
    if usage_stats.total_runtime_hours > 0:
        error_rate = usage_stats.error_count / usage_stats.total_runtime_hours
        error_score = max(0, 100 - (error_rate * 1000))  # Penalize errors
        health_factors.append(error_score * 0.3)  # 30% weight
    else:
        health_factors.append(50.0)  # Neutral score

    # Alert severity factor
    recent_alerts = [a for a in alerts if
                     (datetime.utcnow() - a.timestamp).days <= 7]
    if recent_alerts:
        severity_scores = {'critical': 0, 'high': 25, 'medium': 50, 'low': 75}
        avg_alert_score = sum(
            severity_scores.get(a.severity, 50) for a in recent_alerts) / len(
            recent_alerts)
        health_factors.append(avg_alert_score * 0.2)  # 20% weight
    else:
        health_factors.append(100.0)  # No recent alerts = good

    # Uptime factor
    # Assume device should be available most of the time
    uptime_score = min(100.0, (usage_stats.total_runtime_hours / (
                24 * 7)) * 100)  # Weekly uptime
    health_factors.append(uptime_score * 0.2)  # 20% weight

    return sum(health_factors) / len(
        health_factors) if health_factors else 50.0


def get_device_recommendations(device: DeviceInfo,
                               usage_stats: DeviceUsageStatistics,
                               health_score: float) -> List[str]:
    """Get maintenance and optimization recommendations for a device"""
    recommendations = []

    # Health-based recommendations
    if health_score < 50:
        recommendations.append(
            "Device health is poor. Consider professional inspection.")
    elif health_score < 70:
        recommendations.append(
            "Device performance is declining. Schedule maintenance check.")

    # Usage-based recommendations
    if usage_stats.error_count > 5:
        recommendations.append(
            "High error count detected. Check device configuration.")

    if usage_stats.efficiency_score < 60:
        recommendations.append(
            "Low efficiency detected. Consider energy optimization settings.")

    # Device-specific recommendations
    if device.device_type == 'hvac':
        if usage_stats.total_runtime_hours > 2000:  # High usage
            recommendations.append(
                "HVAC system has high usage. Check air filters.")

    elif device.device_type == 'appliance':
        if usage_stats.cycle_count > 1000:
            recommendations.append(
                "High cycle count. Consider maintenance or replacement.")

    # Age-based recommendations
    if device.installation_date:
        age_years = (
                                datetime.utcnow() - device.installation_date).days / 365.25
        if age_years > 10:
            recommendations.append(
                "Device is over 10 years old. Consider upgrade for efficiency.")
        elif age_years > 5:
            recommendations.append(
                "Device is aging. Monitor performance closely.")

    return recommendations