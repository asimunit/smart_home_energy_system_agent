from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Dict, Any, List
from enum import Enum


class ComfortPriority(Enum):
    COMFORT_FIRST = "comfort_first"
    BALANCED = "balanced"
    EFFICIENCY_FIRST = "efficiency_first"


class ScheduleType(Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    VACATION = "vacation"


@dataclass
class TemperaturePreferences:
    """Temperature comfort preferences"""
    heating_setpoint: float = 70.0  # °F
    cooling_setpoint: float = 76.0  # °F
    tolerance: float = 2.0  # ±°F acceptable range
    setback_temperature: float = 5.0  # °F setback when away
    sleep_setback: float = 3.0  # °F setback during sleep
    auto_adjust_enabled: bool = True
    schedule: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'heating_setpoint': self.heating_setpoint,
            'cooling_setpoint': self.cooling_setpoint,
            'tolerance': self.tolerance,
            'setback_temperature': self.setback_temperature,
            'sleep_setback': self.sleep_setback,
            'auto_adjust_enabled': self.auto_adjust_enabled,
            'schedule': self.schedule
        }


@dataclass
class LightingPreferences:
    """Lighting comfort preferences"""
    brightness_preference: int = 75  # 0-100%
    color_temperature_day: int = 4000  # Kelvin
    color_temperature_evening: int = 2700  # Kelvin
    color_temperature_night: int = 2200  # Kelvin
    auto_adjust_enabled: bool = True
    circadian_lighting: bool = True
    motion_sensor_enabled: bool = True
    daylight_harvesting: bool = True
    scene_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'brightness_preference': self.brightness_preference,
            'color_temperature_day': self.color_temperature_day,
            'color_temperature_evening': self.color_temperature_evening,
            'color_temperature_night': self.color_temperature_night,
            'auto_adjust_enabled': self.auto_adjust_enabled,
            'circadian_lighting': self.circadian_lighting,
            'motion_sensor_enabled': self.motion_sensor_enabled,
            'daylight_harvesting': self.daylight_harvesting,
            'scene_preferences': self.scene_preferences
        }


@dataclass
class HumidityPreferences:
    """Humidity comfort preferences"""
    target_range: Dict[str, float] = field(
        default_factory=lambda: {'min': 40.0, 'max': 60.0})
    tolerance: float = 10.0  # ±% acceptable range
    auto_control_enabled: bool = True
    seasonal_adjustment: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_range': self.target_range,
            'tolerance': self.tolerance,
            'auto_control_enabled': self.auto_control_enabled,
            'seasonal_adjustment': self.seasonal_adjustment
        }


@dataclass
class AirQualityPreferences:
    """Air quality preferences"""
    ventilation_preference: str = "auto"  # auto, manual, timer
    air_circulation_enabled: bool = True
    filter_reminder_enabled: bool = True
    outdoor_air_integration: bool = True
    target_co2_level: int = 1000  # ppm

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ventilation_preference': self.ventilation_preference,
            'air_circulation_enabled': self.air_circulation_enabled,
            'filter_reminder_enabled': self.filter_reminder_enabled,
            'outdoor_air_integration': self.outdoor_air_integration,
            'target_co2_level': self.target_co2_level
        }


@dataclass
class EnergyPreferences:
    """Energy management preferences"""
    priority: ComfortPriority = ComfortPriority.BALANCED
    budget_target: Optional[float] = None  # Monthly $ budget
    efficiency_target: Optional[float] = None  # kWh/sq ft target
    renewable_energy_preference: float = 0.8  # 80% renewable preference
    peak_avoidance_enabled: bool = True
    load_shifting_enabled: bool = True
    demand_response_participation: bool = True
    net_metering_optimization: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'priority': self.priority.value,
            'budget_target': self.budget_target,
            'efficiency_target': self.efficiency_target,
            'renewable_energy_preference': self.renewable_energy_preference,
            'peak_avoidance_enabled': self.peak_avoidance_enabled,
            'load_shifting_enabled': self.load_shifting_enabled,
            'demand_response_participation': self.demand_response_participation,
            'net_metering_optimization': self.net_metering_optimization
        }


@dataclass
class AppliancePreferences:
    """Appliance operation preferences"""
    auto_scheduling_enabled: bool = True
    eco_mode_preference: bool = True
    delay_tolerance: Dict[str, int] = field(default_factory=lambda: {
        'dishwasher': 4,  # hours
        'washing_machine': 6,
        'dryer': 2,
        'oven': 0
    })
    preferred_operating_hours: Dict[str, List[int]] = field(
        default_factory=dict)
    maintenance_reminders: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'auto_scheduling_enabled': self.auto_scheduling_enabled,
            'eco_mode_preference': self.eco_mode_preference,
            'delay_tolerance': self.delay_tolerance,
            'preferred_operating_hours': self.preferred_operating_hours,
            'maintenance_reminders': self.maintenance_reminders
        }


@dataclass
class EVChargingPreferences:
    """Electric vehicle charging preferences"""
    preferred_soc_limit: float = 90.0  # % state of charge
    minimum_departure_soc: float = 80.0  # % minimum for departure
    smart_charging_enabled: bool = True
    solar_charging_preference: bool = True
    departure_schedule: Dict[str, str] = field(default_factory=lambda: {
        'weekday': '08:00',
        'weekend': '10:00'
    })
    return_schedule: Dict[str, str] = field(default_factory=lambda: {
        'weekday': '18:00',
        'weekend': '20:00'
    })
    preconditioning_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'preferred_soc_limit': self.preferred_soc_limit,
            'minimum_departure_soc': self.minimum_departure_soc,
            'smart_charging_enabled': self.smart_charging_enabled,
            'solar_charging_preference': self.solar_charging_preference,
            'departure_schedule': self.departure_schedule,
            'return_schedule': self.return_schedule,
            'preconditioning_enabled': self.preconditioning_enabled
        }


@dataclass
class NotificationPreferences:
    """Notification and alert preferences"""
    energy_alerts_enabled: bool = True
    maintenance_reminders: bool = True
    efficiency_tips: bool = True
    cost_alerts: bool = True
    system_status_updates: bool = True
    emergency_notifications: bool = True
    preferred_channels: List[str] = field(
        default_factory=lambda: ['app', 'email'])
    quiet_hours: Dict[str, str] = field(default_factory=lambda: {
        'start': '22:00',
        'end': '07:00'
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'energy_alerts_enabled': self.energy_alerts_enabled,
            'maintenance_reminders': self.maintenance_reminders,
            'efficiency_tips': self.efficiency_tips,
            'cost_alerts': self.cost_alerts,
            'system_status_updates': self.system_status_updates,
            'emergency_notifications': self.emergency_notifications,
            'preferred_channels': self.preferred_channels,
            'quiet_hours': self.quiet_hours
        }


@dataclass
class OccupancySchedule:
    """Occupancy schedule for optimization"""
    schedule_type: ScheduleType
    schedule_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    zones: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schedule_type': self.schedule_type.value,
            'schedule_data': self.schedule_data,
            'zones': self.zones,
            'enabled': self.enabled
        }


@dataclass
class PrivacyPreferences:
    """Privacy and data preferences"""
    data_sharing_enabled: bool = False
    analytics_participation: bool = True
    learning_enabled: bool = True
    voice_control_enabled: bool = True
    location_services_enabled: bool = True
    cloud_sync_enabled: bool = True
    data_retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_sharing_enabled': self.data_sharing_enabled,
            'analytics_participation': self.analytics_participation,
            'learning_enabled': self.learning_enabled,
            'voice_control_enabled': self.voice_control_enabled,
            'location_services_enabled': self.location_services_enabled,
            'cloud_sync_enabled': self.cloud_sync_enabled,
            'data_retention_days': self.data_retention_days
        }


@dataclass
class UserProfile:
    """Complete user profile with all preferences"""
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "America/New_York"
    language: str = "en"
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Preference categories
    temperature: TemperaturePreferences = field(
        default_factory=TemperaturePreferences)
    lighting: LightingPreferences = field(default_factory=LightingPreferences)
    humidity: HumidityPreferences = field(default_factory=HumidityPreferences)
    air_quality: AirQualityPreferences = field(
        default_factory=AirQualityPreferences)
    energy: EnergyPreferences = field(default_factory=EnergyPreferences)
    appliances: AppliancePreferences = field(
        default_factory=AppliancePreferences)
    ev_charging: EVChargingPreferences = field(
        default_factory=EVChargingPreferences)
    notifications: NotificationPreferences = field(
        default_factory=NotificationPreferences)
    privacy: PrivacyPreferences = field(default_factory=PrivacyPreferences)

    # Schedules
    occupancy_schedules: List[OccupancySchedule] = field(default_factory=list)

    # Custom preferences
    custom_preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'timezone': self.timezone,
            'language': self.language,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'temperature': self.temperature.to_dict(),
            'lighting': self.lighting.to_dict(),
            'humidity': self.humidity.to_dict(),
            'air_quality': self.air_quality.to_dict(),
            'energy': self.energy.to_dict(),
            'appliances': self.appliances.to_dict(),
            'ev_charging': self.ev_charging.to_dict(),
            'notifications': self.notifications.to_dict(),
            'privacy': self.privacy.to_dict(),
            'occupancy_schedules': [schedule.to_dict() for schedule in
                                    self.occupancy_schedules],
            'custom_preferences': self.custom_preferences
        }


@dataclass
class PreferenceChange:
    """Record of preference changes for learning"""
    change_id: str
    user_id: str
    timestamp: datetime
    category: str  # temperature, lighting, etc.
    preference_key: str
    old_value: Any
    new_value: Any
    change_source: str  # user_input, learning, schedule, emergency
    confidence: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'change_id': self.change_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'preference_key': self.preference_key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'change_source': self.change_source,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass
class UserFeedback:
    """User feedback for system learning"""
    feedback_id: str
    user_id: str
    timestamp: datetime
    feedback_type: str  # comfort, efficiency, cost, satisfaction
    category: str  # temperature, lighting, appliance, etc.
    rating: int  # 1-10 scale
    comments: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feedback_id': self.feedback_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'feedback_type': self.feedback_type,
            'category': self.category,
            'rating': self.rating,
            'comments': self.comments,
            'context': self.context,
            'resolved': self.resolved
        }


# Utility functions for user preferences

def create_default_user_profile(user_id: str, name: str) -> UserProfile:
    """Create a default user profile with sensible defaults"""
    return UserProfile(
        user_id=user_id,
        name=name,
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow()
    )


def create_user_profile_from_dict(data: Dict[str, Any]) -> UserProfile:
    """Create UserProfile from dictionary data"""
    profile = UserProfile(
        user_id=data['user_id'],
        name=data['name'],
        email=data.get('email'),
        phone=data.get('phone'),
        timezone=data.get('timezone', 'America/New_York'),
        language=data.get('language', 'en'),
        created_at=datetime.fromisoformat(data['created_at']) if data.get(
            'created_at') else None,
        last_updated=datetime.fromisoformat(data['last_updated']) if data.get(
            'last_updated') else None
    )

    # Load preference categories
    if 'temperature' in data:
        temp_data = data['temperature']
        profile.temperature = TemperaturePreferences(
            heating_setpoint=temp_data.get('heating_setpoint', 70.0),
            cooling_setpoint=temp_data.get('cooling_setpoint', 76.0),
            tolerance=temp_data.get('tolerance', 2.0),
            setback_temperature=temp_data.get('setback_temperature', 5.0),
            sleep_setback=temp_data.get('sleep_setback', 3.0),
            auto_adjust_enabled=temp_data.get('auto_adjust_enabled', True),
            schedule=temp_data.get('schedule', {})
        )

    if 'lighting' in data:
        light_data = data['lighting']
        profile.lighting = LightingPreferences(
            brightness_preference=light_data.get('brightness_preference', 75),
            color_temperature_day=light_data.get('color_temperature_day',
                                                 4000),
            color_temperature_evening=light_data.get(
                'color_temperature_evening', 2700),
            color_temperature_night=light_data.get('color_temperature_night',
                                                   2200),
            auto_adjust_enabled=light_data.get('auto_adjust_enabled', True),
            circadian_lighting=light_data.get('circadian_lighting', True),
            motion_sensor_enabled=light_data.get('motion_sensor_enabled',
                                                 True),
            daylight_harvesting=light_data.get('daylight_harvesting', True),
            scene_preferences=light_data.get('scene_preferences', {})
        )

    if 'energy' in data:
        energy_data = data['energy']
        profile.energy = EnergyPreferences(
            priority=ComfortPriority(energy_data.get('priority', 'balanced')),
            budget_target=energy_data.get('budget_target'),
            efficiency_target=energy_data.get('efficiency_target'),
            renewable_energy_preference=energy_data.get(
                'renewable_energy_preference', 0.8),
            peak_avoidance_enabled=energy_data.get('peak_avoidance_enabled',
                                                   True),
            load_shifting_enabled=energy_data.get('load_shifting_enabled',
                                                  True),
            demand_response_participation=energy_data.get(
                'demand_response_participation', True),
            net_metering_optimization=energy_data.get(
                'net_metering_optimization', True)
        )

    # Load occupancy schedules
    if 'occupancy_schedules' in data:
        for schedule_data in data['occupancy_schedules']:
            schedule = OccupancySchedule(
                schedule_type=ScheduleType(schedule_data['schedule_type']),
                schedule_data=schedule_data.get('schedule_data', {}),
                zones=schedule_data.get('zones', []),
                enabled=schedule_data.get('enabled', True)
            )
            profile.occupancy_schedules.append(schedule)

    profile.custom_preferences = data.get('custom_preferences', {})

    return profile


def validate_user_preferences(profile: UserProfile) -> List[str]:
    """Validate user preferences and return list of validation errors"""
    errors = []

    # Temperature validation
    if profile.temperature.heating_setpoint >= profile.temperature.cooling_setpoint:
        errors.append("Heating setpoint must be lower than cooling setpoint")

    if not (50 <= profile.temperature.heating_setpoint <= 85):
        errors.append("Heating setpoint must be between 50°F and 85°F")

    if not (65 <= profile.temperature.cooling_setpoint <= 95):
        errors.append("Cooling setpoint must be between 65°F and 95°F")

    # Lighting validation
    if not (0 <= profile.lighting.brightness_preference <= 100):
        errors.append("Brightness preference must be between 0 and 100")

    if not (1000 <= profile.lighting.color_temperature_day <= 6500):
        errors.append("Day color temperature must be between 1000K and 6500K")

    # Humidity validation
    humidity_range = profile.humidity.target_range
    if humidity_range['min'] >= humidity_range['max']:
        errors.append("Minimum humidity must be lower than maximum humidity")

    if not (20 <= humidity_range['min'] <= 80):
        errors.append("Minimum humidity must be between 20% and 80%")

    if not (30 <= humidity_range['max'] <= 80):
        errors.append("Maximum humidity must be between 30% and 80%")

    # EV charging validation
    if profile.ev_charging.preferred_soc_limit <= profile.ev_charging.minimum_departure_soc:
        errors.append(
            "Preferred SOC limit must be higher than minimum departure SOC")

    if not (20 <= profile.ev_charging.minimum_departure_soc <= 100):
        errors.append("Minimum departure SOC must be between 20% and 100%")

    return errors


def get_preference_recommendations(profile: UserProfile,
                                   usage_data: Dict[str, Any]) -> List[str]:
    """Generate preference recommendations based on usage patterns"""
    recommendations = []

    # Energy efficiency recommendations
    if profile.energy.priority == ComfortPriority.COMFORT_FIRST:
        avg_consumption = usage_data.get('average_daily_consumption', 0)
        if avg_consumption > 40:  # kWh/day - high consumption
            recommendations.append(
                "Consider switching to 'Balanced' energy priority to reduce consumption while maintaining comfort"
            )

    # Temperature recommendations
    temp_adjustments = usage_data.get('manual_temperature_adjustments', 0)
    if temp_adjustments > 10:  # High number of manual adjustments
        recommendations.append(
            "Frequent temperature adjustments detected. Consider adjusting your temperature preferences or schedule"
        )

    # Lighting recommendations
    if profile.lighting.circadian_lighting and not profile.lighting.auto_adjust_enabled:
        recommendations.append(
            "Enable auto-adjust lighting to maximize circadian lighting benefits"
        )

    # Appliance recommendations
    if not profile.appliances.auto_scheduling_enabled:
        potential_savings = usage_data.get('scheduling_savings_potential', 0)
        if potential_savings > 15:  # More than 15% potential savings
            recommendations.append(
                f"Enable appliance auto-scheduling to save up to {potential_savings:.1f}% on energy costs"
            )

    return recommendations