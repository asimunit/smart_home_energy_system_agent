import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from core.base_agent import BaseAgent
from services.device_service import DeviceService
from config.settings import settings

logger = logging.getLogger(__name__)


class EVChargingAgent(BaseAgent):
    """Agent responsible for managing EV charging for optimal energy efficiency and cost"""

    def __init__(self):
        super().__init__("ev_charging", "EVCharging")
        self.device_service = DeviceService()
        self.charging_stations = {}
        self.charging_schedules = {}
        self.vehicle_data = {}
        self.load_balancing_enabled = True
        self.smart_charging_enabled = True
        self.max_total_charging_power = 7200  # 7.2kW default limit

    async def initialize(self):
        """Initialize the EV charging agent"""
        logger.info("Initializing EV Charging Agent")

        # Load EV charging devices
        await self._load_charging_stations()

        # Initialize vehicle data
        await self._initialize_vehicle_data()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'price_update', 'energy_prediction_updated',
                'solar_generation_forecast',
                'energy_emergency', 'demand_response_event', 'grid_event'
            ]
        )

        # Set up charging optimization
        await self._initialize_charging_optimization()

        logger.info("EV Charging Agent initialized successfully")

    async def execute(self):
        """Main execution logic for EV charging management"""
        try:
            # Monitor charging station states
            await self._monitor_charging_stations()

            # Update vehicle connection status
            await self._update_vehicle_connections()

            # Process charging schedules
            await self._process_charging_schedules()

            # Optimize charging based on pricing and grid conditions
            await self._optimize_charging()

            # Manage load balancing
            if self.load_balancing_enabled:
                await self._manage_load_balancing()

            # Check for completion events
            await self._handle_charging_completion()

        except Exception as e:
            logger.error(f"Error in EV charging execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up EV Charging Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents or users"""
        request_type = data.get('request_type')

        if request_type == 'start_charging':
            await self._handle_start_charging_request(sender, data)
        elif request_type == 'stop_charging':
            await self._handle_stop_charging_request(sender, data)
        elif request_type == 'schedule_charging':
            await self._handle_schedule_charging_request(sender, data)
        elif request_type == 'get_charging_status':
            await self._handle_status_request(sender, data)
        elif request_type == 'set_charging_limit':
            await self._handle_charging_limit_request(sender, data)
        elif request_type == 'optimize_charging':
            await self._handle_optimize_charging_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'price_update':
            await self._handle_price_update(sender, data)
        elif notification_type == 'energy_prediction_updated':
            await self._handle_energy_prediction(sender, data)
        elif notification_type == 'solar_generation_forecast':
            await self._handle_solar_forecast(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)
        elif notification_type == 'demand_response_event':
            await self._handle_demand_response(sender, data)

    async def _load_charging_stations(self):
        """Load EV charging stations from device service"""
        try:
            devices = await self.device_service.get_devices_by_type(
                'ev_charging')

            for device in devices:
                device_id = device['device_id']
                self.charging_stations[device_id] = {
                    'device_info': device,
                    'current_state': None,
                    'connected_vehicle': None,
                    'charging_session': None,
                    'power_limit': device.get('power_rating', 7200),  # Watts
                    'max_current': device.get('max_current', 30),  # Amps
                    'connector_type': device.get('connector_type', 'J1772'),
                    'location': device.get('room', 'garage'),
                    'priority': self._get_station_priority(device),
                    'load_balancing_group': device.get('load_balancing_group',
                                                       'default')
                }

            logger.info(
                f"Loaded {len(self.charging_stations)} EV charging stations")

        except Exception as e:
            logger.error(f"Error loading charging stations: {e}")

    def _get_station_priority(self, device: Dict[str, Any]) -> str:
        """Get priority level for charging station"""
        location = device.get('room', 'garage')

        # Primary garage charger gets highest priority
        if 'garage' in location.lower() and 'main' in device.get('name',
                                                                 '').lower():
            return 'high'
        elif 'garage' in location.lower():
            return 'medium'
        else:
            return 'low'  # Public or secondary chargers

    async def _initialize_vehicle_data(self):
        """Initialize vehicle data and preferences"""
        try:
            # Load vehicle profiles (in production, this would come from user preferences)
            self.vehicle_data = {
                'vehicle_1': {
                    'make': 'Tesla',
                    'model': 'Model 3',
                    'battery_capacity': 75,  # kWh
                    'max_charging_rate': 11.5,  # kW
                    'efficiency': 4.0,  # miles per kWh
                    'charging_curve': 'fast_to_80_then_slow',
                    'preferred_soc_limit': 90,  # State of charge limit
                    'minimum_departure_soc': 80,
                    # Minimum charge for departure
                    'daily_commute_distance': 50,  # miles
                    'departure_schedule': {
                        'weekday': '08:00',
                        'weekend': '10:00'
                    },
                    'return_schedule': {
                        'weekday': '18:00',
                        'weekend': '20:00'
                    }
                }
            }

            logger.info(
                f"Initialized data for {len(self.vehicle_data)} vehicles")

        except Exception as e:
            logger.error(f"Error initializing vehicle data: {e}")

    async def _initialize_charging_optimization(self):
        """Initialize charging optimization parameters"""
        try:
            # Load system constraints and preferences
            self.optimization_params = {
                'prefer_solar_charging': True,
                'avoid_peak_pricing': True,
                'load_balancing_threshold': 0.8,  # 80% of max power
                'minimum_charge_rate': 1.4,  # kW minimum for efficiency
                'smart_departure_buffer': 60,  # minutes before departure
                'cost_optimization_weight': 0.7,  # vs convenience (0.3)
                'grid_friendly_charging': True
            }

            logger.info("Charging optimization parameters initialized")

        except Exception as e:
            logger.error(f"Error initializing charging optimization: {e}")

    async def _monitor_charging_stations(self):
        """Monitor current states of all charging stations"""
        try:
            for station_id, station_data in self.charging_stations.items():
                # Get current device state
                current_state = await self.device_service.get_device_state(
                    station_id)

                if current_state:
                    station_data['current_state'] = current_state

                    # Track energy consumption
                    charging_power = current_state.get('current_power', 0)
                    charging_state = current_state.get('charging_state',
                                                       'idle')

                    # Store consumption data with embedding
                    consumption_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': station_id,
                        'energy_consumption': charging_power / 1000,
                        # Convert W to kW
                        'device_type': 'ev_charging',
                        'charging_state': charging_state,
                        'session_energy': current_state.get('session_energy',
                                                            0),
                        'location': station_data['location']
                    }

                    # Generate embedding for consumption pattern
                    embedding = await self.embedding_service.embed_energy_pattern(
                        consumption_data)
                    consumption_data['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(
                        consumption_data)

                    # Update charging session if active
                    if charging_state == 'charging' and station_data.get(
                            'charging_session'):
                        await self._update_charging_session(station_id,
                                                            current_state)

        except Exception as e:
            logger.error(f"Error monitoring charging stations: {e}")

    async def _update_charging_session(self, station_id: str,
                                       current_state: Dict[str, Any]):
        """Update charging session data"""
        try:
            station_data = self.charging_stations[station_id]
            session = station_data.get('charging_session')

            if session:
                session['current_power'] = current_state.get('current_power',
                                                             0)
                session['session_energy'] = current_state.get('session_energy',
                                                              0)
                session['last_update'] = datetime.utcnow()

                # Calculate charging efficiency
                elapsed_time = (datetime.utcnow() - session[
                    'start_time']).total_seconds() / 3600  # hours
                if elapsed_time > 0:
                    session['average_power'] = session[
                                                   'session_energy'] / elapsed_time

                # Update vehicle SOC estimate
                connected_vehicle = station_data.get('connected_vehicle')
                if connected_vehicle and connected_vehicle in self.vehicle_data:
                    vehicle = self.vehicle_data[connected_vehicle]
                    battery_capacity = vehicle['battery_capacity']

                    # Estimate SOC increase
                    soc_increase = (session[
                                        'session_energy'] / battery_capacity) * 100
                    estimated_soc = session.get('initial_soc',
                                                50) + soc_increase

                    session['estimated_soc'] = min(100, estimated_soc)

        except Exception as e:
            logger.error(f"Error updating charging session: {e}")

    async def _update_vehicle_connections(self):
        """Update vehicle connection status"""
        try:
            for station_id, station_data in self.charging_stations.items():
                current_state = station_data.get('current_state', {})
                connector_state = current_state.get('connector_state',
                                                    'disconnected')

                if connector_state == 'connected':
                    # Vehicle is connected
                    if not station_data.get('connected_vehicle'):
                        # New connection detected
                        await self._handle_vehicle_connection(station_id)

                elif connector_state == 'disconnected':
                    # Vehicle is disconnected
                    if station_data.get('connected_vehicle'):
                        # Disconnection detected
                        await self._handle_vehicle_disconnection(station_id)

        except Exception as e:
            logger.error(f"Error updating vehicle connections: {e}")

    async def _handle_vehicle_connection(self, station_id: str):
        """Handle vehicle connection event"""
        try:
            station_data = self.charging_stations[station_id]

            # In a real system, this would identify the specific vehicle
            # For now, assume it's the default vehicle
            vehicle_id = 'vehicle_1'  # This would be detected/configured

            station_data['connected_vehicle'] = vehicle_id
            station_data['connection_time'] = datetime.utcnow()

            # Get vehicle data
            vehicle = self.vehicle_data.get(vehicle_id, {})

            # Create charging session
            charging_session = {
                'session_id': f"session_{int(datetime.utcnow().timestamp())}",
                'vehicle_id': vehicle_id,
                'station_id': station_id,
                'start_time': datetime.utcnow(),
                'initial_soc': 50,  # Would be read from vehicle
                'target_soc': vehicle.get('preferred_soc_limit', 90),
                'session_energy': 0,
                'current_power': 0,
                'status': 'connected_not_charging'
            }

            station_data['charging_session'] = charging_session

            # Determine if charging should start immediately or be scheduled
            await self._evaluate_charging_start(station_id)

            logger.info(
                f"Vehicle {vehicle_id} connected to station {station_id}")

            # Notify other agents
            await self.broadcast_message(
                'notification',
                {
                    'notification_type': 'ev_connected',
                    'station_id': station_id,
                    'vehicle_id': vehicle_id,
                    'connection_time': datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error handling vehicle connection: {e}")

    async def _handle_vehicle_disconnection(self, station_id: str):
        """Handle vehicle disconnection event"""
        try:
            station_data = self.charging_stations[station_id]
            vehicle_id = station_data.get('connected_vehicle')
            charging_session = station_data.get('charging_session')

            if charging_session:
                # Complete charging session
                charging_session['end_time'] = datetime.utcnow()
                charging_session['status'] = 'completed'

                # Calculate session summary
                duration = (charging_session['end_time'] - charging_session[
                    'start_time']).total_seconds() / 3600
                final_soc = charging_session.get('estimated_soc', 50)

                session_summary = {
                    'session_id': charging_session['session_id'],
                    'vehicle_id': vehicle_id,
                    'duration_hours': duration,
                    'energy_delivered': charging_session['session_energy'],
                    'initial_soc': charging_session.get('initial_soc', 50),
                    'final_soc': final_soc,
                    'average_power': charging_session.get('average_power', 0),
                    'cost_estimate': await self._calculate_session_cost(
                        charging_session)
                }

                # Store session data
                await self.db_service.store_document('device_states', {
                    'timestamp': datetime.utcnow(),
                    'device_id': station_id,
                    'session_type': 'ev_charging_session',
                    'session_data': session_summary
                })

                logger.info(f"Charging session completed: {session_summary}")

            # Clear connection data
            station_data['connected_vehicle'] = None
            station_data['charging_session'] = None
            station_data['connection_time'] = None

            # Notify other agents
            await self.broadcast_message(
                'notification',
                {
                    'notification_type': 'ev_disconnected',
                    'station_id': station_id,
                    'vehicle_id': vehicle_id,
                    'session_summary': session_summary if charging_session else None
                }
            )

        except Exception as e:
            logger.error(f"Error handling vehicle disconnection: {e}")

    async def _evaluate_charging_start(self, station_id: str):
        """Evaluate whether to start charging immediately or schedule it"""
        try:
            station_data = self.charging_stations[station_id]
            vehicle_id = station_data.get('connected_vehicle')
            charging_session = station_data.get('charging_session')

            if not vehicle_id or not charging_session:
                return

            vehicle = self.vehicle_data.get(vehicle_id, {})
            current_soc = charging_session.get('initial_soc', 50)
            target_soc = charging_session.get('target_soc', 90)

            # Check if immediate charging is needed
            minimum_soc = vehicle.get('minimum_departure_soc', 80)
            next_departure = await self._get_next_departure_time(vehicle_id)

            if current_soc < minimum_soc:
                # Critical charge needed
                time_to_departure = (
                                                next_departure - datetime.utcnow()).total_seconds() / 3600  # hours
                energy_needed = ((minimum_soc - current_soc) / 100) * vehicle[
                    'battery_capacity']
                min_charging_time = energy_needed / vehicle[
                    'max_charging_rate']

                if time_to_departure <= min_charging_time * 1.2:  # 20% buffer
                    # Start charging immediately
                    await self._start_charging(station_id, urgent=True)
                    return

            if self.smart_charging_enabled:
                # Schedule optimal charging
                await self._schedule_optimal_charging(station_id)
            else:
                # Start charging immediately
                await self._start_charging(station_id)

        except Exception as e:
            logger.error(f"Error evaluating charging start: {e}")

    async def _get_next_departure_time(self, vehicle_id: str) -> datetime:
        """Get next departure time for vehicle"""
        try:
            vehicle = self.vehicle_data.get(vehicle_id, {})
            departure_schedule = vehicle.get('departure_schedule', {})

            current_time = datetime.utcnow()
            is_weekend = current_time.weekday() >= 5

            schedule_key = 'weekend' if is_weekend else 'weekday'
            departure_time_str = departure_schedule.get(schedule_key, '08:00')

            # Parse departure time
            hour, minute = map(int, departure_time_str.split(':'))

            # Calculate next departure
            next_departure = current_time.replace(hour=hour, minute=minute,
                                                  second=0, microsecond=0)

            # If departure time has passed today, schedule for tomorrow
            if next_departure <= current_time:
                next_departure += timedelta(days=1)

            return next_departure

        except Exception as e:
            logger.error(f"Error getting next departure time: {e}")
            return datetime.utcnow() + timedelta(hours=12)  # Default 12 hours

    async def _schedule_optimal_charging(self, station_id: str):
        """Schedule optimal charging based on pricing and grid conditions"""
        try:
            station_data = self.charging_stations[station_id]
            vehicle_id = station_data.get('connected_vehicle')
            charging_session = station_data.get('charging_session')

            if not vehicle_id or not charging_session:
                return

            vehicle = self.vehicle_data.get(vehicle_id, {})
            current_soc = charging_session.get('initial_soc', 50)
            target_soc = charging_session.get('target_soc', 90)
            next_departure = await self._get_next_departure_time(vehicle_id)

            # Calculate energy requirements
            energy_needed = ((target_soc - current_soc) / 100) * vehicle[
                'battery_capacity']
            max_charging_rate = min(
                vehicle['max_charging_rate'],
                station_data['power_limit'] / 1000  # Convert W to kW
            )

            # Get pricing forecast
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_price_forecast', {'hours': 24}
                )
                price_forecast = price_response.get('price_forecast', {}).get(
                    'predictions', [])
            except:
                price_forecast = []

            # Get solar generation forecast if available
            try:
                solar_response = await self._request_from_agent(
                    'solar_battery', 'get_generation_forecast', {'hours': 24}
                )
                solar_forecast = solar_response.get('generation_forecast', [])
            except:
                solar_forecast = []

            # Create charging schedule using LLM
            optimization_context = {
                'vehicle_id': vehicle_id,
                'current_soc': current_soc,
                'target_soc': target_soc,
                'energy_needed': energy_needed,
                'max_charging_rate': max_charging_rate,
                'departure_time': next_departure.isoformat(),
                'price_forecast': price_forecast[:12],  # Next 12 hours
                'solar_forecast': solar_forecast[:12],
                'preferences': self.optimization_params
            }

            schedule = await self._generate_charging_schedule(
                optimization_context)

            if schedule:
                # Store the schedule
                self.charging_schedules[station_id] = schedule

                logger.info(
                    f"Generated optimal charging schedule for {station_id}")

                # Notify about scheduled charging
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'ev_charging_scheduled',
                        'station_id': station_id,
                        'vehicle_id': vehicle_id,
                        'schedule': schedule,
                        'estimated_cost': schedule.get('estimated_cost', 0),
                        'estimated_completion': schedule.get('completion_time')
                    }
                )

        except Exception as e:
            logger.error(f"Error scheduling optimal charging: {e}")

    async def _generate_charging_schedule(self, context: Dict[str, Any]) -> \
    Dict[str, Any]:
        """Generate optimal charging schedule using LLM"""
        try:
            prompt = f"""
            You are an EV charging optimization expert. Create an optimal charging schedule.

            Vehicle Information:
            - Vehicle: {context['vehicle_id']}
            - Current SOC: {context['current_soc']}%
            - Target SOC: {context['target_soc']}%
            - Energy needed: {context['energy_needed']:.1f} kWh
            - Max charging rate: {context['max_charging_rate']:.1f} kW
            - Departure time: {context['departure_time']}

            Price Forecast (next 12 hours):
            {json.dumps(context['price_forecast'][:6], indent=2)}

            Solar Forecast (if available):
            {json.dumps(context['solar_forecast'][:6], indent=2)}

            Optimization Preferences:
            - Prefer solar charging: {context['preferences']['prefer_solar_charging']}
            - Avoid peak pricing: {context['preferences']['avoid_peak_pricing']}
            - Cost optimization weight: {context['preferences']['cost_optimization_weight']}

            Create a charging schedule that minimizes cost while ensuring the vehicle is ready for departure.

            Respond in JSON format:
            {{
                "charging_periods": [
                    {{
                        "start_time": "YYYY-MM-DDTHH:MM:SS",
                        "end_time": "YYYY-MM-DDTHH:MM:SS", 
                        "charging_rate": <kW>,
                        "reason": "<explanation>"
                    }}
                ],
                "estimated_cost": <dollars>,
                "completion_time": "YYYY-MM-DDTHH:MM:SS",
                "total_energy": <kWh>,
                "confidence": <0-1>
            }}
            """

            response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )

            # Parse the response
            schedule = self.llm_client._parse_json_response(response)

            if 'charging_periods' in schedule:
                return schedule
            else:
                logger.warning("Failed to generate valid charging schedule")
                return None

        except Exception as e:
            logger.error(f"Error generating charging schedule: {e}")
            return None

    async def _start_charging(self, station_id: str, urgent: bool = False):
        """Start charging at a station"""
        try:
            station_data = self.charging_stations[station_id]
            charging_session = station_data.get('charging_session')

            if not charging_session:
                logger.warning(
                    f"No charging session found for station {station_id}")
                return False

            # Determine charging rate
            if urgent:
                # Use maximum available power for urgent charging
                charging_rate = station_data['power_limit']
            else:
                # Use optimal charging rate based on preferences
                charging_rate = await self._calculate_optimal_charging_rate(
                    station_id)

            # Start charging
            success = await self.device_service.set_device_state(
                station_id,
                {
                    'charging_state': 'charging',
                    'current_power': charging_rate
                }
            )

            if success:
                # Update session
                charging_session['status'] = 'charging'
                charging_session['charging_start_time'] = datetime.utcnow()
                charging_session['charging_rate'] = charging_rate

                logger.info(
                    f"Started charging at {station_id} with {charging_rate}W")

                # Notify other agents about power consumption change
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'ev_charging_started',
                        'station_id': station_id,
                        'charging_rate': charging_rate,
                        'urgent': urgent
                    }
                )

                return True
            else:
                logger.error(f"Failed to start charging at {station_id}")
                return False

        except Exception as e:
            logger.error(f"Error starting charging: {e}")
            return False

    async def _calculate_optimal_charging_rate(self, station_id: str) -> int:
        """Calculate optimal charging rate for a station"""
        try:
            station_data = self.charging_stations[station_id]
            vehicle_id = station_data.get('connected_vehicle')

            if not vehicle_id:
                return station_data['power_limit']

            vehicle = self.vehicle_data.get(vehicle_id, {})
            max_vehicle_rate = vehicle.get('max_charging_rate',
                                           11.5) * 1000  # Convert kW to W
            max_station_rate = station_data['power_limit']

            # Base rate is minimum of vehicle and station limits
            base_rate = min(max_vehicle_rate, max_station_rate)

            # Adjust for load balancing if needed
            if self.load_balancing_enabled:
                total_charging_power = sum(
                    station.get('current_state', {}).get('current_power', 0)
                    for station in self.charging_stations.values()
                )

                available_power = self.max_total_charging_power - total_charging_power
                optimal_rate = min(base_rate, available_power)
            else:
                optimal_rate = base_rate

            # Ensure minimum charging rate for efficiency
            min_rate = self.optimization_params[
                           'minimum_charge_rate'] * 1000  # Convert kW to W

            return max(min_rate, optimal_rate)

        except Exception as e:
            logger.error(f"Error calculating optimal charging rate: {e}")
            return 1400  # Default 1.4kW

    async def _process_charging_schedules(self):
        """Process scheduled charging events"""
        try:
            current_time = datetime.utcnow()

            for station_id, schedule in self.charging_schedules.items():
                charging_periods = schedule.get('charging_periods', [])

                for period in charging_periods:
                    start_time = datetime.fromisoformat(period['start_time'])
                    end_time = datetime.fromisoformat(period['end_time'])

                    # Check if we should start charging
                    if (start_time <= current_time <= end_time):
                        station_data = self.charging_stations.get(station_id,
                                                                  {})
                        current_state = station_data.get('current_state', {})

                        if current_state.get('charging_state') != 'charging':
                            # Start scheduled charging
                            charging_rate = period.get('charging_rate',
                                                       3.3) * 1000  # Convert kW to W

                            success = await self.device_service.set_device_state(
                                station_id,
                                {
                                    'charging_state': 'charging',
                                    'current_power': charging_rate
                                }
                            )

                            if success:
                                logger.info(
                                    f"Started scheduled charging at {station_id}: {period['reason']}")

                    # Check if we should stop charging
                    elif current_time > end_time:
                        station_data = self.charging_stations.get(station_id,
                                                                  {})
                        current_state = station_data.get('current_state', {})

                        if current_state.get('charging_state') == 'charging':
                            # Stop scheduled charging
                            success = await self.device_service.set_device_state(
                                station_id,
                                {
                                    'charging_state': 'idle',
                                    'current_power': 0
                                }
                            )

                            if success:
                                logger.info(
                                    f"Stopped scheduled charging at {station_id}")

        except Exception as e:
            logger.error(f"Error processing charging schedules: {e}")

    async def _optimize_charging(self):
        """Optimize ongoing charging based on current conditions"""
        try:
            # Get current pricing
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {}
                )
                current_price = price_response.get('current_price', {})
                price_tier = current_price.get('price_tier', 'standard')
            except:
                price_tier = 'standard'

            # Adjust charging rates based on pricing
            for station_id, station_data in self.charging_stations.items():
                current_state = station_data.get('current_state', {})

                if current_state.get('charging_state') == 'charging':
                    current_power = current_state.get('current_power', 0)

                    # Get vehicle data
                    vehicle_id = station_data.get('connected_vehicle')
                    if not vehicle_id:
                        continue

                    charging_session = station_data.get('charging_session', {})
                    estimated_soc = charging_session.get('estimated_soc', 50)
                    target_soc = charging_session.get('target_soc', 90)

                    # Calculate new optimal rate
                    new_power = await self._calculate_dynamic_charging_rate(
                        station_id, price_tier, estimated_soc, target_soc
                    )

                    # Apply rate adjustment if significant change
                    if abs(new_power - current_power) > 500:  # 0.5kW threshold
                        success = await self.device_service.set_device_state(
                            station_id,
                            {'current_power': new_power}
                        )

                        if success:
                            logger.debug(
                                f"Adjusted charging rate for {station_id}: {new_power}W")

        except Exception as e:
            logger.error(f"Error optimizing charging: {e}")

    async def _calculate_dynamic_charging_rate(self, station_id: str,
                                               price_tier: str,
                                               current_soc: float,
                                               target_soc: float) -> int:
        """Calculate dynamic charging rate based on current conditions"""
        try:
            station_data = self.charging_stations[station_id]
            base_rate = await self._calculate_optimal_charging_rate(station_id)

            # Adjust based on price tier
            if price_tier == 'peak' and self.optimization_params[
                'avoid_peak_pricing']:
                # Reduce charging during peak pricing if not urgent
                if current_soc > 60:  # Have reasonable charge
                    base_rate *= 0.5  # Reduce to 50%
            elif price_tier == 'off_peak':
                # Increase charging during off-peak if beneficial
                if current_soc < target_soc - 10:  # Still need significant charge
                    base_rate *= 1.2  # Increase by 20%

            # Adjust based on SOC (charging curve simulation)
            if current_soc > 80:
                # Slow down charging as battery fills (typical EV behavior)
                soc_factor = 1.0 - ((
                                                current_soc - 80) / 20) * 0.6  # Reduce up to 60% at 100%
                base_rate *= soc_factor

            # Ensure within limits
            max_rate = station_data['power_limit']
            min_rate = self.optimization_params['minimum_charge_rate'] * 1000

            return max(min_rate, min(max_rate, int(base_rate)))

        except Exception as e:
            logger.error(f"Error calculating dynamic charging rate: {e}")
            return 3300  # Default 3.3kW

    async def _manage_load_balancing(self):
        """Manage load balancing across multiple charging stations"""
        try:
            if len(self.charging_stations) <= 1:
                return  # No need for load balancing with single station

            # Calculate total current charging load
            total_power = 0
            active_stations = []

            for station_id, station_data in self.charging_stations.items():
                current_state = station_data.get('current_state', {})
                current_power = current_state.get('current_power', 0)

                if current_power > 0:
                    total_power += current_power
                    active_stations.append(
                        (station_id, current_power, station_data))

            # Check if load balancing is needed
            if total_power > self.max_total_charging_power * \
                    self.optimization_params['load_balancing_threshold']:
                logger.info(
                    f"Load balancing triggered: {total_power}W > {self.max_total_charging_power * 0.8}W")

                # Sort stations by priority
                active_stations.sort(
                    key=lambda x: self._get_station_charging_priority(x[0]),
                    reverse=True)

                # Redistribute power
                available_power = self.max_total_charging_power

                for station_id, current_power, station_data in active_stations:
                    # Calculate fair share based on priority and needs
                    station_priority = self._get_station_charging_priority(
                        station_id)
                    optimal_rate = await self._calculate_optimal_charging_rate(
                        station_id)

                    # Allocate power based on priority
                    if station_priority == 'high':
                        allocated_power = min(optimal_rate, available_power)
                    elif station_priority == 'medium':
                        allocated_power = min(optimal_rate * 0.8,
                                              available_power)
                    else:
                        allocated_power = min(optimal_rate * 0.6,
                                              available_power)

                    # Update station power if changed
                    if abs(allocated_power - current_power) > 200:  # 200W threshold
                        await self.device_service.set_device_state(
                            station_id,
                            {'current_power': int(allocated_power)}
                        )

                        logger.info(
                            f"Load balancing: adjusted {station_id} to {allocated_power}W")

                    available_power -= allocated_power

                    if available_power <= 0:
                        break

        except Exception as e:
            logger.error(f"Error managing load balancing: {e}")

    def _get_station_charging_priority(self, station_id: str) -> str:
        """Get charging priority for a station based on current conditions"""
        try:
            station_data = self.charging_stations[station_id]
            charging_session = station_data.get('charging_session', {})
            vehicle_id = station_data.get('connected_vehicle')

            if not vehicle_id or not charging_session:
                return 'low'

            # Check urgency based on departure time and current SOC
            current_soc = charging_session.get('estimated_soc', 50)
            next_departure = datetime.utcnow() + timedelta(
                hours=8)  # Simplified

            try:
                next_departure = self._get_next_departure_time(vehicle_id)
            except:
                pass

            time_to_departure = (
                                            next_departure - datetime.utcnow()).total_seconds() / 3600

            if current_soc < 30 and time_to_departure < 4:  # Low SOC, leaving soon
                return 'critical'
            elif current_soc < 50 and time_to_departure < 8:  # Medium SOC, reasonable time
                return 'high'
            elif current_soc < 80:  # Normal charging
                return 'medium'
            else:  # Nearly full, topping off
                return 'low'

        except Exception as e:
            logger.error(f"Error getting station priority: {e}")
            return 'medium'

    async def _handle_charging_completion(self):
        """Handle charging completion events"""
        try:
            for station_id, station_data in self.charging_stations.items():
                charging_session = station_data.get('charging_session')
                current_state = station_data.get('current_state', {})

                if charging_session and current_state.get(
                        'charging_state') == 'charging':
                    estimated_soc = charging_session.get('estimated_soc', 50)
                    target_soc = charging_session.get('target_soc', 90)

                    # Check if target SOC reached
                    if estimated_soc >= target_soc:
                        # Stop charging
                        success = await self.device_service.set_device_state(
                            station_id,
                            {
                                'charging_state': 'complete',
                                'current_power': 0
                            }
                        )

                        if success:
                            charging_session['status'] = 'complete'
                            charging_session[
                                'completion_time'] = datetime.utcnow()

                            logger.info(
                                f"Charging completed at {station_id}, SOC: {estimated_soc}%")

                            # Notify completion
                            await self.broadcast_message(
                                'notification',
                                {
                                    'notification_type': 'ev_charging_complete',
                                    'station_id': station_id,
                                    'vehicle_id': station_data.get(
                                        'connected_vehicle'),
                                    'final_soc': estimated_soc,
                                    'completion_time': datetime.utcnow().isoformat()
                                }
                            )

        except Exception as e:
            logger.error(f"Error handling charging completion: {e}")

    async def _calculate_session_cost(self, charging_session: Dict[
        str, Any]) -> float:
        """Calculate cost of charging session"""
        try:
            # This would integrate with pricing data for accurate cost calculation
            # For now, use a simple estimate
            energy_delivered = charging_session.get('session_energy', 0)  # kWh
            average_price = 0.12  # $/kWh - would be calculated from actual pricing during session

            total_cost = energy_delivered * average_price

            return round(total_cost, 2)

        except Exception as e:
            logger.error(f"Error calculating session cost: {e}")
            return 0.0

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

    async def _handle_price_update(self, sender: str, data: Dict[str, Any]):
        """Handle price update notifications"""
        price_tier = data.get('price_tier', 'standard')
        logger.debug(f"Price update received: {price_tier}")

        # Price changes will be handled in the optimization cycle

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Stop all non-essential charging
            for station_id, station_data in self.charging_stations.items():
                charging_session = station_data.get('charging_session', {})
                current_soc = charging_session.get('estimated_soc', 50)
                priority = self._get_station_charging_priority(station_id)

                # Stop charging unless critical
                if priority != 'critical' and current_soc > 40:  # Has reasonable charge
                    await self.device_service.set_device_state(
                        station_id,
                        {
                            'charging_state': 'paused',
                            'current_power': 0
                        }
                    )

                    logger.warning(
                        f"Emergency pause: stopped charging at {station_id}")

    async def _handle_demand_response(self, sender: str, data: Dict[str, Any]):
        """Handle demand response events"""
        event_type = data.get('event_type')
        target_reduction = data.get('recommended_reduction', 20)

        if event_type == 'peak_demand_reduction':
            # Reduce charging power across all stations
            total_current_power = sum(
                station.get('current_state', {}).get('current_power', 0)
                for station in self.charging_stations.values()
            )

            target_power = total_current_power * (1 - target_reduction / 100)

            # Distribute reduction across stations
            for station_id, station_data in self.charging_stations.items():
                current_state = station_data.get('current_state', {})
                current_power = current_state.get('current_power', 0)

                if current_power > 0:
                    new_power = max(1000,
                                    current_power * 0.7)  # Reduce to 70%, min 1kW

                    await self.device_service.set_device_state(
                        station_id,
                        {'current_power': new_power}
                    )

                    logger.info(
                        f"Demand response: reduced {station_id} charging to {new_power}W")