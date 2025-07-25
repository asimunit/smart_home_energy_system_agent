import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from core.base_agent import BaseAgent
from services.device_service import DeviceService
from services.weather_service import WeatherService
from config.settings import settings

logger = logging.getLogger(__name__)


class HVACAgent(BaseAgent):
    """Agent responsible for managing HVAC systems for optimal energy efficiency and comfort"""

    def __init__(self):
        super().__init__("hvac", "HVAC")
        self.device_service = DeviceService()
        self.weather_service = WeatherService()
        self.hvac_devices = {}
        self.temperature_setpoints = {}
        self.comfort_preferences = {}
        self.energy_efficiency_mode = 'balanced'  # balanced, efficiency, comfort
        self.schedule_overrides = {}

    async def initialize(self):
        """Initialize the HVAC agent"""
        logger.info("Initializing HVAC Agent")

        # Load HVAC devices
        await self._load_hvac_devices()

        # Load user comfort preferences
        await self._load_comfort_preferences()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'price_update', 'weather_update', 'occupancy_change',
                'comfort_preference_change', 'energy_emergency'
            ]
        )

        # Initialize temperature control
        await self._initialize_temperature_control()

        logger.info("HVAC Agent initialized successfully")

    async def execute(self):
        """Main execution logic for HVAC management"""
        try:
            # Monitor current conditions
            await self._monitor_conditions()

            # Optimize temperature setpoints
            await self._optimize_temperature_setpoints()

            # Manage HVAC schedules
            await self._manage_hvac_schedules()

            # Monitor energy consumption
            await self._monitor_energy_consumption()

            # Handle comfort vs efficiency balance
            await self._balance_comfort_efficiency()

        except Exception as e:
            logger.error(f"Error in HVAC execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up HVAC Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents"""
        request_type = data.get('request_type')

        if request_type == 'adjust_temperature':
            await self._handle_temperature_adjustment_request(sender, data)
        elif request_type == 'set_schedule':
            await self._handle_schedule_request(sender, data)
        elif request_type == 'get_hvac_status':
            await self._handle_status_request(sender, data)
        elif request_type == 'emergency_shutdown':
            await self._handle_emergency_shutdown(sender, data)
        elif request_type == 'optimize_energy':
            await self._handle_energy_optimization_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'price_update':
            await self._handle_price_update(sender, data)
        elif notification_type == 'weather_update':
            await self._handle_weather_update(sender, data)
        elif notification_type == 'occupancy_change':
            await self._handle_occupancy_change(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)
        elif notification_type == 'demand_response_event':
            await self._handle_demand_response(sender, data)

    async def _load_hvac_devices(self):
        """Load HVAC devices from device service"""
        try:
            devices = await self.device_service.get_devices_by_type('hvac')

            for device in devices:
                device_id = device['device_id']
                self.hvac_devices[device_id] = {
                    'device_info': device,
                    'current_temp': None,
                    'target_temp': None,
                    'mode': 'auto',  # auto, heat, cool, off
                    'fan_speed': 'auto',
                    'power_consumption': 0,
                    'efficiency_rating': device.get('efficiency_rating', 16),
                    # SEER rating
                    'zones': device.get('zones', ['main']),
                    'last_update': None
                }

            logger.info(f"Loaded {len(self.hvac_devices)} HVAC devices")

        except Exception as e:
            logger.error(f"Error loading HVAC devices: {e}")

    async def _load_comfort_preferences(self):
        """Load user comfort preferences"""
        try:
            # Get user preferences from database
            preferences = await self.db_service.get_user_preferences(
                'default_user')

            if preferences:
                comfort_data = preferences.get('preferences', {}).get(
                    'comfort', {})
                self.comfort_preferences = {
                    'temperature_range': comfort_data.get('temperature_range',
                                                          {'min': 68,
                                                           'max': 78}),
                    'humidity_range': comfort_data.get('humidity_range',
                                                       {'min': 30, 'max': 60}),
                    'schedule': comfort_data.get('schedule', {}),
                    'energy_priority': comfort_data.get('energy_priority',
                                                        'balanced'),
                    'comfort_tolerance': comfort_data.get('comfort_tolerance',
                                                          'medium')
                }
            else:
                # Default preferences
                self.comfort_preferences = {
                    'temperature_range': {'min': 68, 'max': 78},
                    'humidity_range': {'min': 30, 'max': 60},
                    'schedule': {
                        'weekday': {
                            'wake': {'time': '06:00', 'temp': 72},
                            'leave': {'time': '08:00', 'temp': 76},
                            'return': {'time': '18:00', 'temp': 72},
                            'sleep': {'time': '22:00', 'temp': 70}
                        },
                        'weekend': {
                            'wake': {'time': '08:00', 'temp': 72},
                            'sleep': {'time': '23:00', 'temp': 70}
                        }
                    },
                    'energy_priority': 'balanced',
                    'comfort_tolerance': 'medium'
                }

            logger.info("Loaded comfort preferences")

        except Exception as e:
            logger.error(f"Error loading comfort preferences: {e}")

    async def _initialize_temperature_control(self):
        """Initialize temperature control for all HVAC devices"""
        try:
            for device_id, device_data in self.hvac_devices.items():
                # Get current device state
                current_state = await self.device_service.get_device_state(
                    device_id)

                if current_state:
                    device_data['current_temp'] = current_state.get(
                        'current_temperature', 72)
                    device_data['target_temp'] = current_state.get(
                        'target_temperature', 72)
                    device_data['mode'] = current_state.get('mode', 'auto')
                    device_data['fan_speed'] = current_state.get('fan_speed',
                                                                 'auto')

                # Set initial temperature setpoint based on preferences and time
                optimal_temp = await self._calculate_optimal_temperature(
                    device_id)
                self.temperature_setpoints[device_id] = optimal_temp

                logger.debug(
                    f"Initialized temperature control for {device_id}: {optimal_temp}°F")

        except Exception as e:
            logger.error(f"Error initializing temperature control: {e}")

    async def _monitor_conditions(self):
        """Monitor current environmental conditions"""
        try:
            # Get current weather data
            weather_data = await self.weather_service.get_current_weather()

            # Update device states
            for device_id, device_data in self.hvac_devices.items():
                current_state = await self.device_service.get_device_state(
                    device_id)

                if current_state:
                    device_data['current_temp'] = current_state.get(
                        'current_temperature', 72)
                    device_data['target_temp'] = current_state.get(
                        'target_temperature', 72)
                    device_data['mode'] = current_state.get('mode', 'auto')
                    device_data['power_consumption'] = current_state.get(
                        'power_consumption', 0)
                    device_data['last_update'] = datetime.utcnow()

                    # Store device state with embedding
                    state_embedding = await self.embedding_service.embed_device_state(
                        {
                            'device_id': device_id,
                            'device_type': 'hvac',
                            'state': current_state,
                            'weather_conditions': weather_data
                        })

                    await self.db_service.store_device_state({
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'device_type': 'hvac',
                        'state': current_state,
                        'state_embedding': state_embedding,
                        'power_consumption': current_state.get(
                            'power_consumption', 0)
                    })

        except Exception as e:
            logger.error(f"Error monitoring conditions: {e}")

    async def _optimize_temperature_setpoints(self):
        """Optimize temperature setpoints for energy efficiency"""
        try:
            for device_id in self.hvac_devices:
                # Calculate optimal temperature considering multiple factors
                optimal_temp = await self._calculate_optimal_temperature(
                    device_id)
                current_setpoint = self.temperature_setpoints.get(device_id,
                                                                  72)

                # Only adjust if change is significant (>1 degree) to avoid frequent changes
                if abs(optimal_temp - current_setpoint) >= 1:
                    # Check if adjustment is needed through negotiation
                    adjustment_request = {
                        'device_id': device_id,
                        'current_setpoint': current_setpoint,
                        'proposed_setpoint': optimal_temp,
                        'reason': 'energy_optimization',
                        'energy_savings_estimate': await self._estimate_energy_savings(
                            device_id, optimal_temp)
                    }

                    # Request negotiation for significant changes
                    await self.send_message(
                        'negotiator',
                        'request',
                        {
                            'request_type': 'negotiate_conflict',
                            'request_data': adjustment_request,
                            'priority': 2,  # Medium priority
                            'required_resources': [device_id],
                            'power_requirement': self.hvac_devices[device_id][
                                'power_consumption']
                        }
                    )

        except Exception as e:
            logger.error(f"Error optimizing temperature setpoints: {e}")

    async def _calculate_optimal_temperature(self, device_id: str) -> float:
        """Calculate optimal temperature for a device"""
        try:
            device_data = self.hvac_devices[device_id]
            current_temp = device_data['current_temp'] or 72

            # Base temperature from schedule
            base_temp = await self._get_scheduled_temperature()

            # Get external factors
            weather_data = await self.weather_service.get_current_weather()
            outdoor_temp = weather_data.get('temperature',
                                            70) if weather_data else 70

            # Get current pricing information
            price_data = {}
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {})
                price_data = price_response.get('current_price', {})
            except:
                pass

            # Prepare context for LLM optimization
            optimization_context = {
                'device_id': device_id,
                'current_temperature': current_temp,
                'base_scheduled_temperature': base_temp,
                'outdoor_temperature': outdoor_temp,
                'weather_conditions': weather_data.get('conditions',
                                                       'clear') if weather_data else 'clear',
                'energy_price': price_data.get('price_per_kwh', 0.12),
                'price_tier': price_data.get('price_tier', 'standard'),
                'comfort_preferences': self.comfort_preferences,
                'device_efficiency': device_data['efficiency_rating'],
                'energy_mode': self.energy_efficiency_mode
            }

            # Generate optimal temperature using LLM
            optimal_temp_response = await self.llm_client.generate_response(
                prompt=self._build_temperature_optimization_prompt(
                    optimization_context),
                temperature=0.3,
                max_tokens=200
            )

            # Parse response
            try:
                # Extract temperature from response
                response_data = json.loads(optimal_temp_response)
                optimal_temp = response_data.get('optimal_temperature',
                                                 base_temp)

                # Ensure temperature is within reasonable bounds
                min_temp = self.comfort_preferences['temperature_range']['min']
                max_temp = self.comfort_preferences['temperature_range']['max']

                # Apply comfort tolerance
                tolerance = {'low': 1, 'medium': 2, 'high': 3}[
                    self.comfort_preferences['comfort_tolerance']]

                if self.energy_efficiency_mode == 'efficiency':
                    # Allow wider range for energy savings
                    min_temp -= tolerance
                    max_temp += tolerance
                elif self.energy_efficiency_mode == 'comfort':
                    # Tighter range for comfort
                    min_temp += tolerance / 2
                    max_temp -= tolerance / 2

                optimal_temp = max(min_temp, min(max_temp, optimal_temp))

                return optimal_temp

            except (json.JSONDecodeError, KeyError):
                logger.warning(
                    f"Failed to parse optimal temperature response for {device_id}")
                return base_temp

        except Exception as e:
            logger.error(
                f"Error calculating optimal temperature for {device_id}: {e}")
            return 72  # Default fallback

    def _build_temperature_optimization_prompt(self,
                                               context: Dict[str, Any]) -> str:
        """Build prompt for temperature optimization"""
        return f"""
        You are an HVAC optimization expert. Given the following context, determine the optimal temperature setpoint.

        Current Situation:
        - Device: {context['device_id']}
        - Current indoor temperature: {context['current_temperature']}°F
        - Scheduled temperature: {context['base_scheduled_temperature']}°F
        - Outdoor temperature: {context['outdoor_temperature']}°F
        - Weather: {context['weather_conditions']}
        - Energy price: ${context['energy_price']}/kWh ({context['price_tier']} rate)
        - Device efficiency: {context['device_efficiency']} SEER
        - Energy mode: {context['energy_mode']}

        User Preferences:
        - Temperature range: {context['comfort_preferences']['temperature_range']['min']}°F - {context['comfort_preferences']['temperature_range']['max']}°F
        - Energy priority: {context['comfort_preferences']['energy_priority']}
        - Comfort tolerance: {context['comfort_preferences']['comfort_tolerance']}

        Consider:
        1. Energy efficiency and cost savings
        2. User comfort preferences
        3. Weather conditions and outdoor temperature
        4. Current energy pricing
        5. HVAC system efficiency

        Respond with JSON format:
        {{
            "optimal_temperature": <temperature_in_fahrenheit>,
            "reasoning": "<brief_explanation>",
            "energy_impact": "<low|medium|high>",
            "comfort_impact": "<minimal|moderate|significant>"
        }}
        """

    async def _get_scheduled_temperature(self) -> float:
        """Get scheduled temperature based on current time and day"""
        try:
            current_time = datetime.utcnow()
            current_hour = current_time.hour
            is_weekend = current_time.weekday() >= 5

            schedule_key = 'weekend' if is_weekend else 'weekday'
            schedule = self.comfort_preferences['schedule'].get(schedule_key,
                                                                {})

            # Find the current period
            current_temp = 72  # Default

            if schedule:
                periods = ['wake', 'leave', 'return', 'sleep']
                for period in periods:
                    if period in schedule:
                        period_time = schedule[period]['time']
                        period_hour = int(period_time.split(':')[0])

                        if current_hour >= period_hour:
                            current_temp = schedule[period]['temp']

            return current_temp

        except Exception as e:
            logger.error(f"Error getting scheduled temperature: {e}")
            return 72

    async def _manage_hvac_schedules(self):
        """Manage HVAC schedules and apply optimizations"""
        try:
            for device_id in self.hvac_devices:
                current_setpoint = self.temperature_setpoints.get(device_id,
                                                                  72)
                device_data = self.hvac_devices[device_id]

                # Check if device needs adjustment
                current_target = device_data.get('target_temp', 72)

                if abs(current_setpoint - current_target) >= 0.5:  # Significant difference
                    # Apply the new setpoint
                    success = await self.device_service.set_device_state(
                        device_id,
                        {'target_temperature': current_setpoint}
                    )

                    if success:
                        device_data['target_temp'] = current_setpoint
                        logger.info(
                            f"Updated {device_id} temperature setpoint to {current_setpoint}°F")
                    else:
                        logger.warning(
                            f"Failed to update temperature setpoint for {device_id}")

        except Exception as e:
            logger.error(f"Error managing HVAC schedules: {e}")

    async def _monitor_energy_consumption(self):
        """Monitor HVAC energy consumption"""
        try:
            total_hvac_consumption = 0

            for device_id, device_data in self.hvac_devices.items():
                consumption = device_data.get('power_consumption', 0)
                total_hvac_consumption += consumption

                # Store individual device consumption
                consumption_data = {
                    'timestamp': datetime.utcnow(),
                    'device_id': device_id,
                    'energy_consumption': consumption / 1000,
                    # Convert W to kW
                    'device_type': 'hvac',
                    'current_temperature': device_data.get('current_temp'),
                    'target_temperature': device_data.get('target_temp'),
                    'mode': device_data.get('mode')
                }

                # Generate embedding for consumption pattern
                embedding = await self.embedding_service.embed_energy_pattern(
                    consumption_data)
                consumption_data['pattern_embedding'] = embedding

                await self.db_service.store_energy_pattern(consumption_data)

            # Check for high consumption alerts
            if total_hvac_consumption > 5000:  # 5kW threshold
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'high_hvac_consumption',
                        'total_consumption': total_hvac_consumption,
                        'severity': 'high' if total_hvac_consumption > 8000 else 'medium',
                        'recommendation': 'Consider adjusting temperature setpoints'
                    }
                )

        except Exception as e:
            logger.error(f"Error monitoring energy consumption: {e}")

    async def _balance_comfort_efficiency(self):
        """Balance comfort and energy efficiency based on current conditions"""
        try:
            # Get current system load and pricing
            price_data = {}
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {})
                price_data = price_response.get('current_price', {})
            except:
                pass

            price_tier = price_data.get('price_tier', 'standard')

            # Adjust energy efficiency mode based on pricing
            if price_tier == 'peak' and self.energy_efficiency_mode != 'efficiency':
                logger.info("Switching to efficiency mode due to peak pricing")
                self.energy_efficiency_mode = 'efficiency'

                # Notify about mode change
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'hvac_mode_change',
                        'new_mode': 'efficiency',
                        'reason': 'peak_pricing',
                        'estimated_savings': '15-25%'
                    }
                )

            elif price_tier == 'off_peak' and self.energy_efficiency_mode == 'efficiency':
                logger.info(
                    "Switching to balanced mode due to off-peak pricing")
                self.energy_efficiency_mode = 'balanced'

        except Exception as e:
            logger.error(f"Error balancing comfort and efficiency: {e}")

    async def _estimate_energy_savings(self, device_id: str,
                                       new_setpoint: float) -> float:
        """Estimate energy savings from temperature adjustment"""
        try:
            device_data = self.hvac_devices[device_id]
            current_setpoint = device_data.get('target_temp', 72)
            current_consumption = device_data.get('power_consumption', 0)

            # Rough estimate: 6-8% savings per degree of setpoint adjustment
            temp_difference = abs(new_setpoint - current_setpoint)

            # Direction matters - cooling setpoint increase or heating setpoint decrease saves energy
            weather_data = await self.weather_service.get_current_weather()
            outdoor_temp = weather_data.get('temperature',
                                            70) if weather_data else 70

            savings_percentage = 0

            if outdoor_temp > 75:  # Cooling season
                if new_setpoint > current_setpoint:  # Raising cooling setpoint saves energy
                    savings_percentage = temp_difference * 7  # 7% per degree
            elif outdoor_temp < 65:  # Heating season
                if new_setpoint < current_setpoint:  # Lowering heating setpoint saves energy
                    savings_percentage = temp_difference * 7  # 7% per degree

            return min(savings_percentage, 30)  # Cap at 30% savings

        except Exception as e:
            logger.error(f"Error estimating energy savings: {e}")
            return 0

    async def _request_from_agent(self, agent_id: str, request_type: str,
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to request data from another agent"""
        request_id = f"req_{datetime.utcnow().timestamp()}"

        await self.send_message(agent_id, 'request', {
            'request_id': request_id,
            'request_type': request_type,
            **data
        })

        # Wait for response (simplified - in production, use proper async handling)
        await asyncio.sleep(1)

        # Get response from message queue
        messages = await self.message_broker.get_messages(self.agent_id,
                                                          max_messages=10)

        for message in messages:
            if (message.get('type') == 'response' and
                    message.get('data', {}).get('request_id') == request_id):
                return message.get('data', {})

        return {}

    async def _handle_temperature_adjustment_request(self, sender: str,
                                                     data: Dict[str, Any]):
        """Handle temperature adjustment request"""
        device_id = data.get('device_id')
        new_temperature = data.get('temperature')
        duration = data.get('duration_minutes', None)

        if device_id in self.hvac_devices and new_temperature:
            # Store override
            if duration:
                self.schedule_overrides[device_id] = {
                    'temperature': new_temperature,
                    'expires_at': datetime.utcnow() + timedelta(
                        minutes=duration),
                    'requester': sender
                }

            # Apply temperature change
            success = await self.device_service.set_device_state(
                device_id,
                {'target_temperature': new_temperature}
            )

            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'success' if success else 'failed',
                'device_id': device_id,
                'new_temperature': new_temperature
            })

        else:
            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'failed',
                'error': 'Invalid device_id or temperature'
            })

    async def _handle_price_update(self, sender: str, data: Dict[str, Any]):
        """Handle price update notifications"""
        new_price = data.get('current_price')
        price_tier = data.get('price_tier')

        logger.debug(f"Price update received: {price_tier} - ${new_price}/kWh")

        # Adjust energy efficiency mode based on price tier
        if price_tier == 'peak':
            self.energy_efficiency_mode = 'efficiency'
        elif price_tier == 'off_peak':
            self.energy_efficiency_mode = 'balanced'

    async def _handle_weather_update(self, sender: str, data: Dict[str, Any]):
        """Handle weather update notifications"""
        temperature = data.get('temperature')
        conditions = data.get('conditions')

        logger.debug(f"Weather update: {temperature}°F, {conditions}")

        # Weather updates may trigger temperature setpoint recalculation
        # This will be handled in the next execution cycle

    async def _handle_occupancy_change(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle occupancy change notifications"""
        occupied = data.get('occupied', True)
        zone = data.get('zone', 'main')

        if not occupied:
            # Implement setback when unoccupied
            for device_id, device_data in self.hvac_devices.items():
                if zone in device_data.get('zones', ['main']):
                    current_setpoint = self.temperature_setpoints.get(
                        device_id, 72)

                    # Apply setback (2-4 degrees depending on season)
                    setback_temp = current_setpoint + 3  # Simplified - should consider heating/cooling

                    self.schedule_overrides[device_id] = {
                        'temperature': setback_temp,
                        'expires_at': datetime.utcnow() + timedelta(hours=4),
                        # 4-hour setback
                        'reason': 'unoccupied'
                    }

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Implement aggressive energy saving
            for device_id in self.hvac_devices:
                current_setpoint = self.temperature_setpoints.get(device_id,
                                                                  72)
                emergency_setpoint = current_setpoint + 5  # 5-degree setback

                await self.device_service.set_device_state(
                    device_id,
                    {'target_temperature': emergency_setpoint}
                )

                logger.warning(
                    f"Emergency setback applied to {device_id}: {emergency_setpoint}°F")

    async def _handle_demand_response(self, sender: str, data: Dict[str, Any]):
        """Handle demand response events"""
        event_type = data.get('event_type')
        target_reduction = data.get('recommended_reduction', 20)  # Percentage

        if event_type == 'peak_demand_reduction':
            # Calculate required temperature adjustment for target reduction
            for device_id, device_data in self.hvac_devices.items():
                current_consumption = device_data.get('power_consumption', 0)
                target_reduction_watts = current_consumption * (
                            target_reduction / 100)

                # Estimate temperature adjustment needed (rough calculation)
                temp_adjustment = target_reduction_watts / (
                            current_consumption / 5) if current_consumption > 0 else 2

                current_setpoint = self.temperature_setpoints.get(device_id,
                                                                  72)
                dr_setpoint = current_setpoint + temp_adjustment

                self.schedule_overrides[device_id] = {
                    'temperature': dr_setpoint,
                    'expires_at': datetime.fromisoformat(data.get('end_time')),
                    'reason': 'demand_response',
                    'incentive': data.get('incentive', 0)
                }

                logger.info(
                    f"Demand response setpoint for {device_id}: {dr_setpoint}°F")