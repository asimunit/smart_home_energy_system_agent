import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import math

from core.base_agent import BaseAgent
from services.device_service import DeviceService
from services.weather_service import WeatherService
from config.settings import settings

logger = logging.getLogger(__name__)


class LightingAgent(BaseAgent):
    """Agent responsible for managing smart lighting systems for energy efficiency and comfort"""

    def __init__(self):
        super().__init__("lighting", "Lighting")
        self.device_service = DeviceService()
        self.weather_service = WeatherService()
        self.lighting_devices = {}
        self.occupancy_data = {}
        self.lighting_schedules = {}
        self.energy_saving_mode = False
        self.daylight_harvesting_enabled = True
        self.circadian_lighting_enabled = True

    async def initialize(self):
        """Initialize the lighting agent"""
        logger.info("Initializing Lighting Agent")

        # Load lighting devices
        await self._load_lighting_devices()

        # Initialize lighting schedules
        await self._initialize_lighting_schedules()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'occupancy_change', 'price_update', 'weather_update',
                'energy_emergency', 'daylight_level_change',
                'user_preference_change'
            ]
        )

        # Set up daylight sensors simulation
        await self._initialize_daylight_monitoring()

        logger.info("Lighting Agent initialized successfully")

    async def execute(self):
        """Main execution logic for lighting management"""
        try:
            # Monitor lighting states
            await self._monitor_lighting_states()

            # Update daylight levels
            await self._update_daylight_levels()

            # Apply daylight harvesting
            if self.daylight_harvesting_enabled:
                await self._apply_daylight_harvesting()

            # Apply circadian lighting
            if self.circadian_lighting_enabled:
                await self._apply_circadian_lighting()

            # Optimize for energy efficiency
            await self._optimize_energy_efficiency()

            # Process scheduled lighting changes
            await self._process_lighting_schedules()

            # Detect and handle idle lighting
            await self._handle_idle_lighting()

        except Exception as e:
            logger.error(f"Error in lighting execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Lighting Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents or users"""
        request_type = data.get('request_type')

        if request_type == 'set_lighting':
            await self._handle_set_lighting_request(sender, data)
        elif request_type == 'get_lighting_status':
            await self._handle_status_request(sender, data)
        elif request_type == 'schedule_lighting':
            await self._handle_schedule_lighting_request(sender, data)
        elif request_type == 'enable_energy_saving':
            await self._handle_energy_saving_request(sender, data)
        elif request_type == 'set_scene':
            await self._handle_scene_request(sender, data)
        elif request_type == 'auto_adjust':
            await self._handle_auto_adjust_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'occupancy_change':
            await self._handle_occupancy_change(sender, data)
        elif notification_type == 'price_update':
            await self._handle_price_update(sender, data)
        elif notification_type == 'weather_update':
            await self._handle_weather_update(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)
        elif notification_type == 'daylight_level_change':
            await self._handle_daylight_change(sender, data)

    async def _load_lighting_devices(self):
        """Load lighting devices from device service"""
        try:
            devices = await self.device_service.get_devices_by_type('lighting')

            for device in devices:
                device_id = device['device_id']
                self.lighting_devices[device_id] = {
                    'device_info': device,
                    'current_state': None,
                    'occupancy_controlled': True,
                    'daylight_harvesting': True,
                    'circadian_enabled': device.get('capabilities', []).count(
                        'color_temperature') > 0,
                    'energy_consumption_pattern': {},
                    'room': device.get('room', 'unknown'),
                    'last_occupancy': None,
                    'optimal_brightness': 80,
                    'optimal_color_temp': 2700
                }

                # Initialize occupancy data for each room
                room = device.get('room', 'unknown')
                if room not in self.occupancy_data:
                    self.occupancy_data[room] = {
                        'occupied': False,
                        'last_activity': None,
                        'occupancy_pattern': [],
                        'auto_off_delay': 15  # minutes
                    }

            logger.info(
                f"Loaded {len(self.lighting_devices)} lighting devices")

        except Exception as e:
            logger.error(f"Error loading lighting devices: {e}")

    async def _initialize_lighting_schedules(self):
        """Initialize lighting schedules"""
        try:
            # Default schedules for different room types
            default_schedules = {
                'living_room': {
                    'weekday': [
                        {'time': '06:30', 'brightness': 40,
                         'color_temp': 3000},
                        {'time': '18:00', 'brightness': 80,
                         'color_temp': 2700},
                        {'time': '22:00', 'brightness': 30,
                         'color_temp': 2200},
                        {'time': '23:30', 'brightness': 0, 'color_temp': 2200}
                    ],
                    'weekend': [
                        {'time': '08:00', 'brightness': 50,
                         'color_temp': 2800},
                        {'time': '19:00', 'brightness': 80,
                         'color_temp': 2700},
                        {'time': '23:00', 'brightness': 20,
                         'color_temp': 2200},
                        {'time': '00:30', 'brightness': 0, 'color_temp': 2200}
                    ]
                },
                'bedroom': {
                    'weekday': [
                        {'time': '06:00', 'brightness': 20,
                         'color_temp': 3500},
                        {'time': '21:00', 'brightness': 40,
                         'color_temp': 2200},
                        {'time': '22:30', 'brightness': 0, 'color_temp': 2200}
                    ],
                    'weekend': [
                        {'time': '08:00', 'brightness': 30,
                         'color_temp': 3000},
                        {'time': '22:00', 'brightness': 30,
                         'color_temp': 2200},
                        {'time': '23:30', 'brightness': 0, 'color_temp': 2200}
                    ]
                },
                'kitchen': {
                    'weekday': [
                        {'time': '06:00', 'brightness': 90,
                         'color_temp': 4000},
                        {'time': '08:30', 'brightness': 60,
                         'color_temp': 3500},
                        {'time': '17:30', 'brightness': 90,
                         'color_temp': 3500},
                        {'time': '21:00', 'brightness': 40,
                         'color_temp': 2700},
                        {'time': '23:00', 'brightness': 0, 'color_temp': 2700}
                    ],
                    'weekend': [
                        {'time': '08:00', 'brightness': 80,
                         'color_temp': 3500},
                        {'time': '12:00', 'brightness': 70,
                         'color_temp': 3500},
                        {'time': '18:00', 'brightness': 90,
                         'color_temp': 3500},
                        {'time': '22:00', 'brightness': 30,
                         'color_temp': 2700},
                        {'time': '00:00', 'brightness': 0, 'color_temp': 2700}
                    ]
                }
            }

            # Apply schedules to devices
            for device_id, device_data in self.lighting_devices.items():
                room = device_data['room']
                room_type = self._classify_room_type(room)

                if room_type in default_schedules:
                    self.lighting_schedules[device_id] = default_schedules[
                        room_type]
                else:
                    # Default generic schedule
                    self.lighting_schedules[device_id] = default_schedules[
                        'living_room']

            logger.info("Lighting schedules initialized")

        except Exception as e:
            logger.error(f"Error initializing lighting schedules: {e}")

    def _classify_room_type(self, room_name: str) -> str:
        """Classify room type from room name"""
        room_lower = room_name.lower()

        if 'bedroom' in room_lower or 'bed' in room_lower:
            return 'bedroom'
        elif 'kitchen' in room_lower:
            return 'kitchen'
        elif 'living' in room_lower or 'family' in room_lower:
            return 'living_room'
        elif 'bathroom' in room_lower or 'bath' in room_lower:
            return 'bathroom'
        elif 'office' in room_lower or 'study' in room_lower:
            return 'office'
        else:
            return 'living_room'  # Default

    async def _initialize_daylight_monitoring(self):
        """Initialize daylight monitoring system"""
        try:
            # Get current weather data for initial daylight calculation
            weather_data = await self.weather_service.get_current_weather()

            for device_id, device_data in self.lighting_devices.items():
                room = device_data['room']

                # Simulate daylight sensors for each room
                daylight_level = await self._calculate_daylight_level(room,
                                                                      weather_data)

                device_data['daylight_level'] = daylight_level
                device_data['last_daylight_update'] = datetime.utcnow()

            logger.info("Daylight monitoring initialized")

        except Exception as e:
            logger.error(f"Error initializing daylight monitoring: {e}")

    async def _calculate_daylight_level(self, room: str,
                                        weather_data: Dict[str, Any]) -> float:
        """Calculate daylight level for a room"""
        try:
            current_time = datetime.utcnow()
            hour = current_time.hour

            # Base daylight calculation
            if 6 <= hour <= 18:  # Daylight hours
                # Peak daylight at noon
                sun_position = math.sin((hour - 6) * math.pi / 12)
                base_light = sun_position * 100  # 0-100 scale
            else:
                base_light = 0

            # Adjust for weather conditions
            if weather_data:
                conditions = weather_data.get('conditions', 'clear')
                weather_factors = {
                    'clear': 1.0,
                    'partly_cloudy': 0.7,
                    'cloudy': 0.4,
                    'overcast': 0.2,
                    'rainy': 0.1,
                    'foggy': 0.3
                }

                weather_factor = weather_factors.get(conditions, 0.7)
                base_light *= weather_factor

            # Adjust for room characteristics
            room_factors = {
                'living_room': 0.8,  # Large windows
                'kitchen': 0.9,  # Multiple windows
                'bedroom': 0.6,  # Medium windows
                'bathroom': 0.3,  # Small/no windows
                'hallway': 0.2,  # Limited natural light
                'basement': 0.1  # Minimal natural light
            }

            room_type = self._classify_room_type(room)
            room_factor = room_factors.get(room_type, 0.7)

            final_light_level = base_light * room_factor

            return max(0, min(100, final_light_level))  # Clamp to 0-100

        except Exception as e:
            logger.error(f"Error calculating daylight level: {e}")
            return 50  # Default moderate level

    async def _monitor_lighting_states(self):
        """Monitor current states of all lighting devices"""
        try:
            for device_id, device_data in self.lighting_devices.items():
                # Get current device state
                current_state = await self.device_service.get_device_state(
                    device_id)

                if current_state:
                    device_data['current_state'] = current_state

                    # Track energy consumption
                    consumption = current_state.get('power_consumption', 0)

                    # Store consumption data with embedding
                    consumption_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'energy_consumption': consumption / 1000,
                        # Convert W to kW
                        'device_type': 'lighting',
                        'room': device_data['room'],
                        'brightness': current_state.get('brightness', 0),
                        'color_temperature': current_state.get(
                            'color_temperature', 2700)
                    }

                    # Generate embedding for consumption pattern
                    embedding = await self.embedding_service.embed_energy_pattern(
                        consumption_data)
                    consumption_data['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(
                        consumption_data)

                    # Update consumption pattern
                    await self._update_consumption_pattern(device_id,
                                                           consumption_data)

        except Exception as e:
            logger.error(f"Error monitoring lighting states: {e}")

    async def _update_consumption_pattern(self, device_id: str,
                                          consumption_data: Dict[str, Any]):
        """Update consumption pattern for a lighting device"""
        try:
            device_data = self.lighting_devices[device_id]
            pattern = device_data['energy_consumption_pattern']

            brightness = consumption_data.get('brightness', 0)
            consumption = consumption_data.get('energy_consumption', 0)

            # Categorize by brightness level
            if brightness == 0:
                category = 'off'
            elif brightness <= 25:
                category = 'low'
            elif brightness <= 75:
                category = 'medium'
            else:
                category = 'high'

            if category not in pattern:
                pattern[category] = {
                    'total_consumption': 0,
                    'sample_count': 0,
                    'average_consumption': 0
                }

            pattern[category]['total_consumption'] += consumption
            pattern[category]['sample_count'] += 1
            pattern[category]['average_consumption'] = (
                    pattern[category]['total_consumption'] /
                    pattern[category]['sample_count']
            )

        except Exception as e:
            logger.error(
                f"Error updating consumption pattern for {device_id}: {e}")

    async def _update_daylight_levels(self):
        """Update daylight levels for all rooms"""
        try:
            weather_data = await self.weather_service.get_current_weather()

            for device_id, device_data in self.lighting_devices.items():
                room = device_data['room']

                # Update daylight level
                new_daylight_level = await self._calculate_daylight_level(room,
                                                                          weather_data)
                old_daylight_level = device_data.get('daylight_level', 0)

                device_data['daylight_level'] = new_daylight_level
                device_data['last_daylight_update'] = datetime.utcnow()

                # Notify about significant changes
                if abs(new_daylight_level - old_daylight_level) > 20:  # 20% change
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'daylight_level_change',
                            'room': room,
                            'old_level': old_daylight_level,
                            'new_level': new_daylight_level,
                            'change': new_daylight_level - old_daylight_level
                        }
                    )

        except Exception as e:
            logger.error(f"Error updating daylight levels: {e}")

    async def _apply_daylight_harvesting(self):
        """Apply daylight harvesting to reduce artificial lighting"""
        try:
            for device_id, device_data in self.lighting_devices.items():
                if not device_data.get('daylight_harvesting', True):
                    continue

                current_state = device_data.get('current_state', {})
                current_brightness = current_state.get('brightness', 0)
                daylight_level = device_data.get('daylight_level', 0)
                room = device_data['room']

                # Check if room is occupied
                occupancy = self.occupancy_data.get(room, {})
                if not occupancy.get('occupied', False):
                    continue

                # Calculate optimal brightness considering daylight
                target_illumination = 80  # Target illumination level

                if daylight_level > target_illumination:
                    # Sufficient daylight, turn off artificial lighting
                    optimal_brightness = 0
                elif daylight_level > 20:
                    # Some daylight, reduce artificial lighting
                    optimal_brightness = max(0,
                                             target_illumination - daylight_level)
                else:
                    # Low daylight, use full artificial lighting as needed
                    optimal_brightness = device_data.get('optimal_brightness',
                                                         80)

                # Apply gradual adjustment to avoid sudden changes
                if abs(optimal_brightness - current_brightness) > 10:
                    # Gradual adjustment (10% per cycle)
                    if optimal_brightness > current_brightness:
                        new_brightness = min(optimal_brightness,
                                             current_brightness + 10)
                    else:
                        new_brightness = max(optimal_brightness,
                                             current_brightness - 10)

                    # Update device brightness
                    success = await self.device_service.set_device_state(
                        device_id,
                        {'brightness': new_brightness}
                    )

                    if success:
                        logger.debug(
                            f"Daylight harvesting: adjusted {device_id} brightness to {new_brightness}%")

        except Exception as e:
            logger.error(f"Error applying daylight harvesting: {e}")

    async def _apply_circadian_lighting(self):
        """Apply circadian lighting based on time of day"""
        try:
            current_time = datetime.utcnow()
            hour = current_time.hour

            # Calculate optimal color temperature based on time
            if 6 <= hour < 9:  # Morning
                target_color_temp = 4000  # Cool white for alertness
            elif 9 <= hour < 15:  # Day
                target_color_temp = 5000  # Daylight white
            elif 15 <= hour < 18:  # Afternoon
                target_color_temp = 4000  # Cool white
            elif 18 <= hour < 21:  # Evening
                target_color_temp = 3000  # Warm white
            else:  # Night
                target_color_temp = 2200  # Very warm white for relaxation

            for device_id, device_data in self.lighting_devices.items():
                if not device_data.get('circadian_enabled', False):
                    continue

                current_state = device_data.get('current_state', {})
                current_color_temp = current_state.get('color_temperature',
                                                       2700)
                current_brightness = current_state.get('brightness', 0)

                # Only adjust if light is on and room is occupied
                room = device_data['room']
                occupancy = self.occupancy_data.get(room, {})

                if current_brightness > 0 and occupancy.get('occupied', False):
                    # Gradual color temperature adjustment
                    if abs(target_color_temp - current_color_temp) > 200:
                        if target_color_temp > current_color_temp:
                            new_color_temp = min(target_color_temp,
                                                 current_color_temp + 200)
                        else:
                            new_color_temp = max(target_color_temp,
                                                 current_color_temp - 200)

                        # Update device color temperature
                        success = await self.device_service.set_device_state(
                            device_id,
                            {'color_temperature': new_color_temp}
                        )

                        if success:
                            device_data[
                                'optimal_color_temp'] = target_color_temp
                            logger.debug(
                                f"Circadian lighting: adjusted {device_id} color temp to {new_color_temp}K")

        except Exception as e:
            logger.error(f"Error applying circadian lighting: {e}")

    async def _optimize_energy_efficiency(self):
        """Optimize lighting for energy efficiency"""
        try:
            # Get current pricing information
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {}
                )
                price_tier = price_response.get('current_price', {}).get(
                    'price_tier', 'standard')
            except:
                price_tier = 'standard'

            # Apply energy saving measures during peak pricing
            if price_tier == 'peak' or self.energy_saving_mode:
                for device_id, device_data in self.lighting_devices.items():
                    current_state = device_data.get('current_state', {})
                    current_brightness = current_state.get('brightness', 0)
                    room = device_data['room']

                    # Check if room is occupied
                    occupancy = self.occupancy_data.get(room, {})
                    if not occupancy.get('occupied', False):
                        # Turn off lights in unoccupied rooms
                        if current_brightness > 0:
                            await self.device_service.set_device_state(
                                device_id, {'brightness': 0})
                            logger.info(
                                f"Energy optimization: turned off {device_id} (unoccupied)")

                    elif current_brightness > 60:  # Reduce brightness in occupied rooms
                        new_brightness = max(40,
                                             current_brightness * 0.8)  # 20% reduction, min 40%

                        await self.device_service.set_device_state(
                            device_id,
                            {'brightness': new_brightness}
                        )
                        logger.info(
                            f"Energy optimization: reduced {device_id} brightness to {new_brightness}%")

        except Exception as e:
            logger.error(f"Error optimizing energy efficiency: {e}")

    async def _process_lighting_schedules(self):
        """Process scheduled lighting changes"""
        try:
            current_time = datetime.utcnow()
            current_hour_minute = current_time.strftime('%H:%M')
            is_weekend = current_time.weekday() >= 5

            schedule_type = 'weekend' if is_weekend else 'weekday'

            for device_id, schedule in self.lighting_schedules.items():
                room_schedule = schedule.get(schedule_type, [])

                for scheduled_change in room_schedule:
                    scheduled_time = scheduled_change['time']

                    # Check if it's time for this scheduled change (within 1 minute)
                    if self._is_time_match(current_hour_minute,
                                           scheduled_time):
                        device_data = self.lighting_devices.get(device_id, {})
                        room = device_data.get('room', 'unknown')
                        occupancy = self.occupancy_data.get(room, {})

                        # Only apply schedule if room is occupied or it's an 'off' command
                        if occupancy.get('occupied', False) or \
                                scheduled_change['brightness'] == 0:
                            new_state = {
                                'brightness': scheduled_change['brightness'],
                                'color_temperature': scheduled_change.get(
                                    'color_temp', 2700)
                            }

                            success = await self.device_service.set_device_state(
                                device_id, new_state)

                            if success:
                                logger.info(
                                    f"Applied scheduled change to {device_id}: {new_state}")

        except Exception as e:
            logger.error(f"Error processing lighting schedules: {e}")

    def _is_time_match(self, current_time: str, scheduled_time: str) -> bool:
        """Check if current time matches scheduled time (within 1 minute)"""
        try:
            current_parts = current_time.split(':')
            scheduled_parts = scheduled_time.split(':')

            current_minutes = int(current_parts[0]) * 60 + int(
                current_parts[1])
            scheduled_minutes = int(scheduled_parts[0]) * 60 + int(
                scheduled_parts[1])

            # Match if within 1 minute
            return abs(current_minutes - scheduled_minutes) <= 1

        except Exception:
            return False

    async def _handle_idle_lighting(self):
        """Handle lights that have been on in unoccupied rooms"""
        try:
            current_time = datetime.utcnow()

            for device_id, device_data in self.lighting_devices.items():
                current_state = device_data.get('current_state', {})
                current_brightness = current_state.get('brightness', 0)
                room = device_data['room']

                if current_brightness > 0:  # Light is on
                    occupancy = self.occupancy_data.get(room, {})
                    last_activity = occupancy.get('last_activity')
                    auto_off_delay = occupancy.get('auto_off_delay',
                                                   15)  # minutes

                    if last_activity:
                        time_since_activity = (
                                                          current_time - last_activity).total_seconds() / 60

                        if time_since_activity > auto_off_delay:
                            # Turn off idle lighting
                            success = await self.device_service.set_device_state(
                                device_id,
                                {'brightness': 0}
                            )

                            if success:
                                logger.info(
                                    f"Auto-off: turned off idle lighting in {room}")

                                # Update occupancy status
                                occupancy['occupied'] = False

        except Exception as e:
            logger.error(f"Error handling idle lighting: {e}")

    async def _request_from_agent(self, agent_id: str, request_type: str,
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to request data from another agent"""
        request_id = f"req_{datetime.utcnow().timestamp()}"

        await self.send_message(agent_id, 'request', {
            'request_id': request_id,
            'request_type': request_type,
            **data
        })

        # Wait for response (simplified)
        await asyncio.sleep(1)

        # Get response from message queue
        messages = await self.message_broker.get_messages(self.agent_id,
                                                          max_messages=10)

        for message in messages:
            if (message.get('type') == 'response' and
                    message.get('data', {}).get('request_id') == request_id):
                return message.get('data', {})

        return {}

    async def _handle_occupancy_change(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle occupancy change notifications"""
        room = data.get('room', 'unknown')
        occupied = data.get('occupied', False)

        if room in self.occupancy_data:
            self.occupancy_data[room]['occupied'] = occupied
            self.occupancy_data[room]['last_activity'] = datetime.utcnow()

            logger.info(
                f"Occupancy change: {room} is {'occupied' if occupied else 'unoccupied'}")

            # Find lighting devices in this room
            room_devices = [
                device_id for device_id, device_data in
                self.lighting_devices.items()
                if device_data['room'] == room
            ]

            if occupied:
                # Turn on lights when room becomes occupied
                for device_id in room_devices:
                    device_data = self.lighting_devices[device_id]
                    optimal_brightness = device_data.get('optimal_brightness',
                                                         80)
                    optimal_color_temp = device_data.get('optimal_color_temp',
                                                         2700)

                    # Consider daylight level
                    daylight_level = device_data.get('daylight_level', 0)
                    if daylight_level > 60:  # High daylight
                        optimal_brightness = max(20,
                                                 optimal_brightness - daylight_level)

                    await self.device_service.set_device_state(device_id, {
                        'brightness': optimal_brightness,
                        'color_temperature': optimal_color_temp
                    })

            # Note: We let the idle handler turn off lights when unoccupied

    async def _handle_price_update(self, sender: str, data: Dict[str, Any]):
        """Handle price update notifications"""
        price_tier = data.get('price_tier', 'standard')

        if price_tier == 'peak':
            self.energy_saving_mode = True
            logger.info("Enabling energy saving mode due to peak pricing")
        elif price_tier == 'off_peak':
            self.energy_saving_mode = False
            logger.info("Disabling energy saving mode due to off-peak pricing")

    async def _handle_weather_update(self, sender: str, data: Dict[str, Any]):
        """Handle weather update notifications"""
        # Weather updates will trigger daylight level recalculation in the next cycle
        logger.debug(
            "Weather update received, will recalculate daylight levels")

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Turn off all non-essential lighting
            for device_id, device_data in self.lighting_devices.items():
                room = device_data['room']

                # Keep minimal lighting in occupied rooms
                occupancy = self.occupancy_data.get(room, {})
                if occupancy.get('occupied', False):
                    await self.device_service.set_device_state(device_id, {
                        'brightness': 20,  # Minimal lighting
                        'color_temperature': 2200
                    })
                else:
                    await self.device_service.set_device_state(device_id, {
                        'brightness': 0})

                logger.warning(
                    f"Emergency lighting reduction applied to {device_id}")

    async def _handle_set_lighting_request(self, sender: str,
                                           data: Dict[str, Any]):
        """Handle request to set lighting"""
        device_id = data.get('device_id')
        room = data.get('room')
        brightness = data.get('brightness')
        color_temp = data.get('color_temperature')

        target_devices = []

        if device_id and device_id in self.lighting_devices:
            target_devices = [device_id]
        elif room:
            target_devices = [
                dev_id for dev_id, dev_data in self.lighting_devices.items()
                if dev_data['room'] == room
            ]

        success_count = 0
        for target_device in target_devices:
            new_state = {}
            if brightness is not None:
                new_state['brightness'] = brightness
            if color_temp is not None:
                new_state['color_temperature'] = color_temp

            if new_state:
                success = await self.device_service.set_device_state(
                    target_device, new_state)
                if success:
                    success_count += 1

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'status': 'success' if success_count > 0 else 'failed',
            'devices_updated': success_count,
            'total_devices': len(target_devices)
        })

    async def _handle_scene_request(self, sender: str, data: Dict[str, Any]):
        """Handle scene setting request"""
        scene_name = data.get('scene')
        room = data.get('room')

        # Predefined scenes
        scenes = {
            'bright': {'brightness': 100, 'color_temperature': 4000},
            'dim': {'brightness': 30, 'color_temperature': 2700},
            'warm': {'brightness': 70, 'color_temperature': 2200},
            'cool': {'brightness': 80, 'color_temperature': 5000},
            'off': {'brightness': 0}
        }

        if scene_name in scenes:
            scene_settings = scenes[scene_name]

            # Apply to room or all devices
            if room:
                target_devices = [
                    dev_id for dev_id, dev_data in
                    self.lighting_devices.items()
                    if dev_data['room'] == room
                ]
            else:
                target_devices = list(self.lighting_devices.keys())

            success_count = 0
            for device_id in target_devices:
                success = await self.device_service.set_device_state(device_id,
                                                                     scene_settings)
                if success:
                    success_count += 1

            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'success' if success_count > 0 else 'failed',
                'scene': scene_name,
                'devices_updated': success_count
            })
        else:
            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'failed',
                'error': f'Unknown scene: {scene_name}'
            })