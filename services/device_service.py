import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import random

logger = logging.getLogger(__name__)


class DeviceService:
    """Service for interacting with smart home devices"""

    def __init__(self):
        self.devices = {}
        self.device_states = {}
        self.device_types = {
            'hvac': ['thermostat', 'heat_pump', 'air_conditioner', 'furnace'],
            'appliance': ['dishwasher', 'washing_machine', 'dryer',
                          'refrigerator', 'oven'],
            'lighting': ['smart_bulb', 'smart_switch', 'dimmer', 'led_strip'],
            'ev_charging': ['ev_charger', 'level2_charger', 'dc_fast_charger'],
            'solar_battery': ['solar_panel', 'battery_storage', 'inverter'],
            'security': ['security_system', 'smart_lock', 'camera'],
            'entertainment': ['smart_tv', 'sound_system', 'gaming_console']
        }
        self._initialize_mock_devices()
        logger.info("Device service initialized")

    def _initialize_mock_devices(self):
        """Initialize mock devices for demonstration"""
        # HVAC devices
        self.devices['hvac_main'] = {
            'device_id': 'hvac_main',
            'device_type': 'hvac',
            'device_subtype': 'thermostat',
            'name': 'Main Floor Thermostat',
            'room': 'living_room',
            'power_rating': 3500,  # Watts
            'efficiency_rating': 16,  # SEER
            'zones': ['main'],
            'capabilities': ['heating', 'cooling', 'fan', 'humidity_control'],
            'monitoring_enabled': True,
            'controllable': True
        }

        self.devices['hvac_upstairs'] = {
            'device_id': 'hvac_upstairs',
            'device_type': 'hvac',
            'device_subtype': 'thermostat',
            'name': 'Upstairs Thermostat',
            'room': 'master_bedroom',
            'power_rating': 2500,
            'efficiency_rating': 18,
            'zones': ['upstairs'],
            'capabilities': ['heating', 'cooling', 'fan'],
            'monitoring_enabled': True,
            'controllable': True
        }

        # Appliances
        self.devices['dishwasher_main'] = {
            'device_id': 'dishwasher_main',
            'device_type': 'appliance',
            'device_subtype': 'dishwasher',
            'name': 'Kitchen Dishwasher',
            'room': 'kitchen',
            'power_rating': 1800,
            'energy_star_rating': 'A+',
            'capabilities': ['normal_wash', 'eco_wash', 'quick_wash',
                             'delay_start'],
            'monitoring_enabled': True,
            'controllable': True
        }

        self.devices['washer_main'] = {
            'device_id': 'washer_main',
            'device_type': 'appliance',
            'device_subtype': 'washing_machine',
            'name': 'Laundry Room Washer',
            'room': 'laundry_room',
            'power_rating': 2000,
            'capabilities': ['normal', 'delicate', 'heavy_duty', 'quick_wash',
                             'delay_start'],
            'monitoring_enabled': True,
            'controllable': True
        }

        self.devices['dryer_main'] = {
            'device_id': 'dryer_main',
            'device_type': 'appliance',
            'device_subtype': 'dryer',
            'name': 'Laundry Room Dryer',
            'room': 'laundry_room',
            'power_rating': 3000,
            'capabilities': ['normal', 'delicate', 'heavy_duty', 'air_dry',
                             'delay_start'],
            'monitoring_enabled': True,
            'controllable': True
        }

        # Lighting
        for room in ['living_room', 'kitchen', 'master_bedroom',
                     'guest_bedroom', 'bathroom']:
            device_id = f'lights_{room}'
            self.devices[device_id] = {
                'device_id': device_id,
                'device_type': 'lighting',
                'device_subtype': 'smart_bulb',
                'name': f'{room.replace("_", " ").title()} Lights',
                'room': room,
                'power_rating': 60,  # 60W equivalent LED
                'capabilities': ['on_off', 'dimming', 'color_temperature'],
                'monitoring_enabled': True,
                'controllable': True
            }

        # EV Charging
        self.devices['ev_charger_garage'] = {
            'device_id': 'ev_charger_garage',
            'device_type': 'ev_charging',
            'device_subtype': 'level2_charger',
            'name': 'Garage EV Charger',
            'room': 'garage',
            'power_rating': 7200,  # 7.2kW Level 2 charger
            'max_current': 30,  # Amps
            'capabilities': ['level2_charging', 'scheduling',
                             'load_balancing'],
            'monitoring_enabled': True,
            'controllable': True
        }

        # Solar and Battery
        self.devices['solar_panels'] = {
            'device_id': 'solar_panels',
            'device_type': 'solar_battery',
            'device_subtype': 'solar_panel',
            'name': 'Rooftop Solar Array',
            'room': 'roof',
            'power_rating': 8000,  # 8kW system
            'panel_count': 24,
            'capabilities': ['power_generation', 'monitoring'],
            'monitoring_enabled': True,
            'controllable': False
        }

        self.devices['battery_storage'] = {
            'device_id': 'battery_storage',
            'device_type': 'solar_battery',
            'device_subtype': 'battery_storage',
            'name': 'Home Battery System',
            'room': 'garage',
            'power_rating': 5000,  # 5kW continuous power
            'capacity': 13500,  # 13.5kWh capacity
            'capabilities': ['energy_storage', 'backup_power',
                             'load_shifting'],
            'monitoring_enabled': True,
            'controllable': True
        }

        # Initialize device states
        for device_id, device_info in self.devices.items():
            self.device_states[device_id] = self._generate_initial_state(
                device_info)

    def _generate_initial_state(self, device_info: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate initial state for a device"""
        device_type = device_info['device_type']
        device_subtype = device_info.get('device_subtype', '')

        base_state = {
            'power_state': 'on',
            'power_consumption': 0,
            'last_updated': datetime.utcnow().isoformat(),
            'online': True,
            'error_state': None
        }

        if device_type == 'hvac':
            base_state.update({
                'current_temperature': random.uniform(70, 75),
                'target_temperature': 72,
                'mode': 'auto',  # auto, cool, heat, off
                'fan_speed': 'auto',  # auto, low, medium, high
                'humidity': random.uniform(35, 55),
                'power_consumption': random.uniform(0, device_info[
                    'power_rating'])
            })

        elif device_type == 'appliance':
            if device_subtype == 'dishwasher':
                base_state.update({
                    'cycle_state': 'idle',  # idle, washing, drying, complete
                    'cycle_type': 'normal',
                    'time_remaining': 0,
                    'door_open': False,
                    'power_consumption': 0
                })
            elif device_subtype in ['washing_machine', 'dryer']:
                base_state.update({
                    'cycle_state': 'idle',  # idle, running, complete
                    'cycle_type': 'normal',
                    'time_remaining': 0,
                    'door_locked': False,
                    'power_consumption': 0
                })

        elif device_type == 'lighting':
            base_state.update({
                'brightness': 80,  # 0-100%
                'color_temperature': 2700,  # Kelvin
                'color': None,  # RGB if color bulb
                'power_consumption': device_info['power_rating'] * 0.8
                # 80% brightness
            })

        elif device_type == 'ev_charging':
            base_state.update({
                'charging_state': 'idle',  # idle, charging, complete, error
                'current_power': 0,
                'session_energy': 0,
                'connector_state': 'disconnected',  # connected, disconnected
                'power_consumption': 0
            })

        elif device_type == 'solar_battery':
            if device_subtype == 'solar_panel':
                base_state.update({
                    'power_generation': random.uniform(0, device_info[
                        'power_rating'] * 0.3),  # 30% of max
                    'daily_generation': random.uniform(20, 40),  # kWh
                    'panel_temperature': random.uniform(70, 90),
                    'irradiance': random.uniform(200, 800)  # W/m²
                })
            elif device_subtype == 'battery_storage':
                base_state.update({
                    'state_of_charge': random.uniform(40, 90),  # 40-90%
                    'power_flow': random.uniform(-1000, 1000),
                    # Negative = charging, Positive = discharging
                    'available_capacity': device_info['capacity'] * 0.9,
                    # 90% available
                    'power_consumption': abs(random.uniform(-1000, 1000))
                })

        return base_state

    async def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all devices"""
        return list(self.devices.values())

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific device by ID"""
        return self.devices.get(device_id)

    async def get_devices_by_type(self, device_type: str) -> List[
        Dict[str, Any]]:
        """Get all devices of a specific type"""
        return [device for device in self.devices.values()
                if device['device_type'] == device_type]

    async def get_devices_by_room(self, room: str) -> List[Dict[str, Any]]:
        """Get all devices in a specific room"""
        return [device for device in self.devices.values()
                if device.get('room') == room]

    async def get_device_state(self, device_id: str) -> Optional[
        Dict[str, Any]]:
        """Get current state of a device"""
        if device_id not in self.devices:
            return None

        # Simulate state updates
        await self._update_device_state(device_id)
        return self.device_states.get(device_id)

    async def set_device_state(self, device_id: str,
                               new_state: Dict[str, Any]) -> bool:
        """Set device state"""
        try:
            if device_id not in self.devices:
                logger.error(f"Device not found: {device_id}")
                return False

            device_info = self.devices[device_id]
            if not device_info.get('controllable', False):
                logger.error(f"Device not controllable: {device_id}")
                return False

            # Update device state
            current_state = self.device_states.get(device_id, {})
            current_state.update(new_state)
            current_state['last_updated'] = datetime.utcnow().isoformat()

            # Update power consumption based on new state
            await self._calculate_power_consumption(device_id, current_state)

            self.device_states[device_id] = current_state

            logger.debug(f"Updated state for {device_id}: {new_state}")
            return True

        except Exception as e:
            logger.error(f"Error setting device state for {device_id}: {e}")
            return False

    async def get_device_consumption(self, device_id: str) -> Optional[
        Dict[str, Any]]:
        """Get current power consumption for a device"""
        state = await self.get_device_state(device_id)
        if state:
            return {
                'device_id': device_id,
                'power_consumption': state.get('power_consumption', 0),
                'timestamp': state.get('last_updated'),
                'power_state': state.get('power_state', 'unknown')
            }
        return None

    async def start_device_cycle(self, device_id: str, cycle_type: str,
                                 delay_minutes: int = 0) -> bool:
        """Start a device cycle (for appliances)"""
        try:
            device_info = self.devices.get(device_id)
            if not device_info or device_info['device_type'] != 'appliance':
                return False

            current_state = self.device_states.get(device_id, {})

            # Set cycle parameters
            cycle_durations = {
                'dishwasher': {'normal': 120, 'eco': 180, 'quick': 60},
                'washing_machine': {'normal': 45, 'delicate': 30,
                                    'heavy_duty': 60, 'quick': 20},
                'dryer': {'normal': 60, 'delicate': 40, 'heavy_duty': 80,
                          'air_dry': 120}
            }

            subtype = device_info.get('device_subtype', '')
            duration = cycle_durations.get(subtype, {}).get(cycle_type, 60)

            # Update state
            current_state.update({
                'cycle_state': 'running',
                'cycle_type': cycle_type,
                'time_remaining': duration + delay_minutes,
                'start_time': (datetime.utcnow() if delay_minutes == 0 else
                               datetime.utcnow()).isoformat()
            })

            self.device_states[device_id] = current_state

            logger.info(
                f"Started {cycle_type} cycle for {device_id}, duration: {duration} minutes")
            return True

        except Exception as e:
            logger.error(f"Error starting device cycle for {device_id}: {e}")
            return False

    async def schedule_device_operation(self, device_id: str,
                                        operation: Dict[str, Any],
                                        start_time: datetime) -> bool:
        """Schedule a device operation"""
        try:
            # In a real implementation, this would interface with the device's scheduling system
            # For now, we'll simulate by storing the schedule

            if device_id not in self.devices:
                return False

            # Store scheduled operation (simplified)
            current_state = self.device_states.get(device_id, {})

            if 'scheduled_operations' not in current_state:
                current_state['scheduled_operations'] = []

            current_state['scheduled_operations'].append({
                'operation': operation,
                'start_time': start_time.isoformat(),
                'status': 'scheduled'
            })

            self.device_states[device_id] = current_state

            logger.info(f"Scheduled operation for {device_id} at {start_time}")
            return True

        except Exception as e:
            logger.error(
                f"Error scheduling device operation for {device_id}: {e}")
            return False

    async def get_device_capabilities(self, device_id: str) -> List[str]:
        """Get device capabilities"""
        device_info = self.devices.get(device_id)
        if device_info:
            return device_info.get('capabilities', [])
        return []

    async def get_energy_usage_history(self, device_id: str,
                                       hours: int = 24) -> List[
        Dict[str, Any]]:
        """Get energy usage history for a device"""
        # Simulate historical data
        history = []
        current_time = datetime.utcnow()

        device_info = self.devices.get(device_id)
        if not device_info:
            return history

        # Generate mock historical data
        for i in range(hours):
            timestamp = current_time - timedelta(hours=i)

            # Simulate varying power consumption
            base_power = device_info.get('power_rating', 100)
            variation = random.uniform(0.7, 1.3)  # ±30% variation
            power_consumption = base_power * variation

            # Apply usage patterns based on device type
            hour = timestamp.hour
            if device_info['device_type'] == 'hvac':
                # HVAC typically higher during peak hours
                if 16 <= hour <= 21:  # Peak hours
                    power_consumption *= 1.5
                elif 22 <= hour <= 6:  # Night setback
                    power_consumption *= 0.6

            elif device_info['device_type'] == 'lighting':
                # Lights typically on during evening/night
                if 6 <= hour <= 22:
                    power_consumption = base_power
                else:
                    power_consumption = 0

            history.append({
                'timestamp': timestamp.isoformat(),
                'power_consumption': power_consumption,
                'energy_consumption': power_consumption / 1000,  # kWh
                'device_state': 'on' if power_consumption > 0 else 'off'
            })

        return history

    async def _update_device_state(self, device_id: str):
        """Update device state with realistic variations"""
        device_info = self.devices.get(device_id)
        current_state = self.device_states.get(device_id, {})

        if not device_info or not current_state:
            return

        device_type = device_info['device_type']

        # Update based on device type
        if device_type == 'hvac':
            await self._update_hvac_state(device_id, current_state)
        elif device_type == 'appliance':
            await self._update_appliance_state(device_id, current_state)
        elif device_type == 'solar_battery':
            await self._update_solar_battery_state(device_id, current_state)

        # Update power consumption
        await self._calculate_power_consumption(device_id, current_state)

        current_state['last_updated'] = datetime.utcnow().isoformat()
        self.device_states[device_id] = current_state

    async def _update_hvac_state(self, device_id: str, state: Dict[str, Any]):
        """Update HVAC device state"""
        # Simulate temperature changes
        current_temp = state.get('current_temperature', 72)
        target_temp = state.get('target_temperature', 72)

        # Simple temperature convergence simulation
        temp_diff = target_temp - current_temp
        if abs(temp_diff) > 0.1:
            # Move towards target temperature
            temp_change = temp_diff * 0.1  # 10% of difference per update
            state['current_temperature'] = current_temp + temp_change

        # Update humidity (simulate)
        state['humidity'] = max(30, min(70, state.get('humidity',
                                                      45) + random.uniform(-2,
                                                                           2)))

    async def _update_appliance_state(self, device_id: str,
                                      state: Dict[str, Any]):
        """Update appliance device state"""
        if state.get('cycle_state') == 'running':
            time_remaining = state.get('time_remaining', 0)
            if time_remaining > 0:
                # Decrease remaining time
                state['time_remaining'] = max(0,
                                              time_remaining - 1)  # Decrease by 1 minute per update

                if state['time_remaining'] == 0:
                    state['cycle_state'] = 'complete'
                    logger.info(f"Appliance cycle completed for {device_id}")

    async def _update_solar_battery_state(self, device_id: str,
                                          state: Dict[str, Any]):
        """Update solar and battery device state"""
        device_info = self.devices[device_id]
        subtype = device_info.get('device_subtype', '')

        if subtype == 'solar_panel':
            # Simulate solar generation based on time of day
            current_hour = datetime.utcnow().hour
            max_power = device_info['power_rating']

            # Simple solar curve (peak at noon)
            if 6 <= current_hour <= 18:  # Daylight hours
                solar_factor = 0.5 * (1 + cos((current_hour - 12) * pi / 6))
                generation = max_power * solar_factor * random.uniform(0.8,
                                                                       1.0)
            else:
                generation = 0

            state['power_generation'] = generation

        elif subtype == 'battery_storage':
            # Simulate battery state changes
            current_soc = state.get('state_of_charge', 50)
            power_flow = state.get('power_flow', 0)

            # Simple battery simulation
            if power_flow < 0:  # Charging
                new_soc = min(100, current_soc + abs(
                    power_flow) / 1000)  # Simplified
            elif power_flow > 0:  # Discharging
                new_soc = max(0, current_soc - power_flow / 1000)  # Simplified
            else:
                new_soc = current_soc

            state['state_of_charge'] = new_soc

    async def _calculate_power_consumption(self, device_id: str,
                                           state: Dict[str, Any]):
        """Calculate current power consumption for a device"""
        device_info = self.devices[device_id]
        device_type = device_info['device_type']
        power_rating = device_info.get('power_rating', 0)

        if state.get('power_state') == 'off':
            state['power_consumption'] = 0
            return

        if device_type == 'hvac':
            # HVAC consumption varies based on temperature differential and mode
            current_temp = state.get('current_temperature', 72)
            target_temp = state.get('target_temperature', 72)
            temp_diff = abs(current_temp - target_temp)

            # Higher consumption when temperature difference is larger
            consumption_factor = min(1.0, 0.3 + (temp_diff * 0.1))
            state['power_consumption'] = power_rating * consumption_factor

        elif device_type == 'appliance':
            cycle_state = state.get('cycle_state', 'idle')
            if cycle_state == 'running':
                # Full power during cycle
                state['power_consumption'] = power_rating
            else:
                # Standby power
                state['power_consumption'] = power_rating * 0.02  # 2% standby

        elif device_type == 'lighting':
            brightness = state.get('brightness', 100) / 100.0
            state['power_consumption'] = power_rating * brightness

        elif device_type == 'ev_charging':
            charging_state = state.get('charging_state', 'idle')
            if charging_state == 'charging':
                current_power = state.get('current_power', power_rating)
                state['power_consumption'] = current_power
            else:
                state['power_consumption'] = 0

        elif device_type == 'solar_battery':
            subtype = device_info.get('device_subtype', '')
            if subtype == 'solar_panel':
                # Solar panels don't consume power, they generate it
                state['power_consumption'] = 0
            elif subtype == 'battery_storage':
                # Battery consumption depends on charging/discharging
                power_flow = state.get('power_flow', 0)
                if power_flow < 0:  # Charging
                    state['power_consumption'] = abs(
                        power_flow) * 1.1  # 10% inefficiency
                else:
                    state['power_consumption'] = 0

        else:
            # Default calculation
            state['power_consumption'] = power_rating * random.uniform(0.5,
                                                                       1.0)


# Import dependencies at the end to avoid circular imports
from datetime import timedelta
from math import cos, pi