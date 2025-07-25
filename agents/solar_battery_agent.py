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


class SolarBatteryAgent(BaseAgent):
    """Agent responsible for managing solar panels and battery storage for optimal energy management"""

    def __init__(self):
        super().__init__("solar_battery", "SolarBattery")
        self.device_service = DeviceService()
        self.weather_service = WeatherService()
        self.solar_devices = {}
        self.battery_devices = {}
        self.generation_forecast = []
        self.storage_strategy = 'balanced'  # balanced, maximize_self_consumption, grid_support
        self.net_metering_enabled = True
        self.backup_power_reserved = 20  # Percent of battery reserved for backup

    async def initialize(self):
        """Initialize the solar battery agent"""
        logger.info("Initializing Solar Battery Agent")

        # Load solar and battery devices
        await self._load_solar_battery_devices()

        # Initialize generation forecasting
        await self._initialize_generation_forecasting()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'weather_update', 'price_update', 'energy_emergency',
                'demand_response_event', 'grid_outage',
                'high_consumption_detected'
            ]
        )

        # Set up optimization parameters
        await self._initialize_optimization_parameters()

        logger.info("Solar Battery Agent initialized successfully")

    async def execute(self):
        """Main execution logic for solar battery management"""
        try:
            # Monitor solar generation and battery state
            await self._monitor_solar_battery_systems()

            # Update generation forecasts
            await self._update_generation_forecast()

            # Optimize energy storage and dispatch
            await self._optimize_energy_storage()

            # Manage grid interaction
            await self._manage_grid_interaction()

            # Handle backup power management
            await self._manage_backup_power()

            # Detect and respond to system anomalies
            await self._detect_system_anomalies()

        except Exception as e:
            logger.error(f"Error in solar battery execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Solar Battery Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents"""
        request_type = data.get('request_type')

        if request_type == 'get_generation_forecast':
            await self._handle_generation_forecast_request(sender, data)
        elif request_type == 'get_battery_status':
            await self._handle_battery_status_request(sender, data)
        elif request_type == 'set_storage_strategy':
            await self._handle_storage_strategy_request(sender, data)
        elif request_type == 'charge_battery':
            await self._handle_charge_battery_request(sender, data)
        elif request_type == 'discharge_battery':
            await self._handle_discharge_battery_request(sender, data)
        elif request_type == 'backup_power_test':
            await self._handle_backup_power_test(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'weather_update':
            await self._handle_weather_update(sender, data)
        elif notification_type == 'price_update':
            await self._handle_price_update(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)
        elif notification_type == 'demand_response_event':
            await self._handle_demand_response(sender, data)
        elif notification_type == 'grid_outage':
            await self._handle_grid_outage(sender, data)
        elif notification_type == 'high_consumption_detected':
            await self._handle_high_consumption(sender, data)

    async def _load_solar_battery_devices(self):
        """Load solar and battery devices from device service"""
        try:
            devices = await self.device_service.get_devices_by_type(
                'solar_battery')

            for device in devices:
                device_id = device['device_id']
                device_subtype = device.get('device_subtype', '')

                if device_subtype == 'solar_panel':
                    self.solar_devices[device_id] = {
                        'device_info': device,
                        'current_state': None,
                        'capacity_kw': device.get('power_rating', 8000) / 1000,
                        # Convert W to kW
                        'panel_count': device.get('panel_count', 24),
                        'efficiency': device.get('efficiency', 0.20),
                        # 20% efficiency
                        'tilt_angle': device.get('tilt_angle', 30),
                        'azimuth': device.get('azimuth', 180),  # South-facing
                        'location': device.get('room', 'roof'),
                        'generation_history': [],
                        'performance_ratio': 0.85  # System losses
                    }

                elif device_subtype == 'battery_storage':
                    self.battery_devices[device_id] = {
                        'device_info': device,
                        'current_state': None,
                        'capacity_kwh': device.get('capacity', 13500) / 1000,
                        # Convert Wh to kWh
                        'max_power_kw': device.get('power_rating',
                                                   5000) / 1000,
                        # Convert W to kW
                        'efficiency': device.get('efficiency', 0.90),
                        # 90% round-trip efficiency
                        'chemistry': device.get('chemistry', 'lithium_ion'),
                        'cycle_count': 0,
                        'location': device.get('room', 'garage'),
                        'backup_capable': device.get('backup_capable', True),
                        'grid_tied': device.get('grid_tied', True)
                    }

            logger.info(
                f"Loaded {len(self.solar_devices)} solar devices and {len(self.battery_devices)} battery devices")

        except Exception as e:
            logger.error(f"Error loading solar battery devices: {e}")

    async def _initialize_generation_forecasting(self):
        """Initialize solar generation forecasting system"""
        try:
            # Get initial weather data for forecasting
            weather_data = await self.weather_service.get_current_weather()

            # Generate initial forecast
            await self._generate_solar_forecast(weather_data)

            logger.info("Solar generation forecasting initialized")

        except Exception as e:
            logger.error(f"Error initializing generation forecasting: {e}")

    async def _initialize_optimization_parameters(self):
        """Initialize optimization parameters"""
        try:
            self.optimization_params = {
                'self_consumption_priority': 0.8,
                # Prioritize self-consumption
                'grid_export_threshold': 0.9,  # Export when battery > 90%
                'battery_charge_threshold': 0.2,  # Charge when battery < 20%
                'backup_reserve_soc': self.backup_power_reserved / 100,
                'peak_shaving_enabled': True,
                'arbitrage_enabled': True,  # Buy low, sell high
                'grid_support_enabled': True,
                'weather_prediction_weight': 0.7,
                'price_prediction_weight': 0.3
            }

            logger.info("Optimization parameters initialized")

        except Exception as e:
            logger.error(f"Error initializing optimization parameters: {e}")

    async def _monitor_solar_battery_systems(self):
        """Monitor current states of solar and battery systems"""
        try:
            # Monitor solar devices
            for device_id, device_data in self.solar_devices.items():
                current_state = await self.device_service.get_device_state(
                    device_id)

                if current_state:
                    device_data['current_state'] = current_state

                    # Track generation
                    generation = current_state.get('power_generation',
                                                   0)  # Watts

                    # Store generation data with embedding
                    generation_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'power_generation': generation / 1000,
                        # Convert W to kW
                        'device_type': 'solar_panel',
                        'daily_generation': current_state.get(
                            'daily_generation', 0),  # kWh
                        'panel_temperature': current_state.get(
                            'panel_temperature', 70),
                        'irradiance': current_state.get('irradiance', 0)
                        # W/m²
                    }

                    # Generate embedding for generation pattern
                    embedding = await self.embedding_service.embed_energy_pattern(
                        generation_data)
                    generation_data['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(generation_data)

                    # Update generation history
                    device_data['generation_history'].append({
                        'timestamp': datetime.utcnow(),
                        'generation': generation / 1000,  # kW
                        'irradiance': current_state.get('irradiance', 0)
                    })

                    # Keep only recent history (last 100 readings)
                    if len(device_data['generation_history']) > 100:
                        device_data['generation_history'] = device_data[
                                                                'generation_history'][
                                                            -100:]

            # Monitor battery devices
            for device_id, device_data in self.battery_devices.items():
                current_state = await self.device_service.get_device_state(
                    device_id)

                if current_state:
                    device_data['current_state'] = current_state

                    # Track battery state
                    soc = current_state.get('state_of_charge', 50)  # Percent
                    power_flow = current_state.get('power_flow',
                                                   0)  # Watts (+ discharging, - charging)

                    # Store battery data with embedding
                    battery_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'state_of_charge': soc,
                        'power_flow': power_flow / 1000,  # Convert W to kW
                        'device_type': 'battery_storage',
                        'available_capacity': current_state.get(
                            'available_capacity', 0) / 1000,  # kWh
                        'cycle_count': device_data.get('cycle_count', 0)
                    }

                    # Generate embedding for battery pattern
                    embedding = await self.embedding_service.embed_energy_pattern(
                        battery_data)
                    battery_data['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(battery_data)

        except Exception as e:
            logger.error(f"Error monitoring solar battery systems: {e}")

    async def _update_generation_forecast(self):
        """Update solar generation forecast"""
        try:
            # Get weather forecast
            weather_forecast = await self.weather_service.get_solar_forecast(
                hours=48)

            if weather_forecast:
                await self._generate_solar_forecast_from_weather(
                    weather_forecast)

        except Exception as e:
            logger.error(f"Error updating generation forecast: {e}")

    async def _generate_solar_forecast_from_weather(self,
                                                    weather_forecast: List[
                                                        Dict[str, Any]]):
        """Generate solar forecast from weather forecast"""
        try:
            self.generation_forecast = []

            for weather_hour in weather_forecast:
                forecast_time = datetime.fromisoformat(
                    weather_hour['datetime'])
                irradiance = weather_hour.get('solar_irradiance', 0)  # W/m²
                temperature = weather_hour.get('temperature', 75)  # °F
                cloud_cover = weather_hour.get('cloud_cover', 0)  # Percent

                # Calculate total system generation
                total_generation = 0

                for device_id, device_data in self.solar_devices.items():
                    # Calculate generation for this solar array
                    capacity_kw = device_data['capacity_kw']
                    efficiency = device_data['efficiency']
                    performance_ratio = device_data['performance_ratio']

                    # Temperature derating (panels lose efficiency when hot)
                    temp_coefficient = -0.004  # -0.4% per °C above 25°C
                    temp_celsius = (temperature - 32) * 5 / 9
                    temp_derating = 1 + temp_coefficient * (temp_celsius - 25)

                    # Calculate generation
                    if irradiance > 0:
                        # Standard Test Conditions: 1000 W/m²
                        irradiance_factor = irradiance / 1000

                        generation_kw = (capacity_kw * irradiance_factor *
                                         efficiency * performance_ratio * temp_derating)

                        # Apply cloud cover reduction
                        cloud_factor = 1 - (
                                    cloud_cover / 100) * 0.7  # 70% reduction at full cloud
                        generation_kw *= cloud_factor

                        total_generation += max(0, generation_kw)

                forecast_entry = {
                    'datetime': forecast_time.isoformat(),
                    'generation_kw': round(total_generation, 2),
                    'irradiance': irradiance,
                    'cloud_cover': cloud_cover,
                    'temperature': temperature,
                    'confidence': weather_hour.get('confidence', 0.8)
                }

                self.generation_forecast.append(forecast_entry)

            logger.debug(
                f"Generated solar forecast for {len(self.generation_forecast)} hours")

            # Notify other agents about updated forecast
            await self.broadcast_message(
                'notification',
                {
                    'notification_type': 'solar_generation_forecast',
                    'forecast': self.generation_forecast[:24],  # Next 24 hours
                    'total_expected_generation_24h': sum(
                        f['generation_kw'] for f in
                        self.generation_forecast[:24]
                    )
                }
            )

        except Exception as e:
            logger.error(f"Error generating solar forecast: {e}")

    async def _optimize_energy_storage(self):
        """Optimize energy storage and dispatch strategy"""
        try:
            # Get current system state
            total_generation = sum(
                device.get('current_state', {}).get('power_generation', 0)
                for device in self.solar_devices.values()
            ) / 1000  # Convert W to kW

            total_battery_soc = 0
            total_battery_capacity = 0
            battery_power_available = 0

            for device_data in self.battery_devices.values():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)
                capacity = device_data['capacity_kwh']
                max_power = device_data['max_power_kw']

                total_battery_soc += soc * capacity
                total_battery_capacity += capacity
                battery_power_available += max_power

            average_soc = total_battery_soc / total_battery_capacity if total_battery_capacity > 0 else 50

            # Get current home energy consumption
            try:
                consumption_response = await self._request_from_agent(
                    'energy_monitor', 'get_consumption_data', {'hours': 1}
                )
                recent_consumption = consumption_response.get(
                    'consumption_data', [])
                current_consumption = sum(
                    data.get('energy_consumption', 0) for data in
                    recent_consumption[-5:]
                ) / max(1, len(recent_consumption[
                               -5:]))  # Average of last 5 readings in kW
            except:
                current_consumption = 3.0  # Default 3kW

            # Get current pricing
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {}
                )
                current_price = price_response.get('current_price', {}).get(
                    'price_per_kwh', 0.12)
                price_tier = price_response.get('current_price', {}).get(
                    'price_tier', 'standard')
            except:
                current_price = 0.12
                price_tier = 'standard'

            # Calculate net energy flow
            net_energy = total_generation - current_consumption  # kW

            # Determine optimal strategy
            strategy = await self._determine_optimal_strategy(
                net_energy, average_soc, price_tier, current_price
            )

            # Execute strategy
            await self._execute_storage_strategy(strategy)

        except Exception as e:
            logger.error(f"Error optimizing energy storage: {e}")

    async def _determine_optimal_strategy(self, net_energy: float,
                                          battery_soc: float,
                                          price_tier: str,
                                          current_price: float) -> Dict[
        str, Any]:
        """Determine optimal energy storage strategy"""
        try:
            # Build context for LLM decision
            context = {
                'net_energy_kw': net_energy,
                'battery_soc_percent': battery_soc,
                'price_tier': price_tier,
                'current_price_per_kwh': current_price,
                'backup_reserve_soc': self.backup_power_reserved,
                'storage_strategy': self.storage_strategy,
                'optimization_params': self.optimization_params,
                'generation_forecast_6h': self.generation_forecast[:6],
                'net_metering_enabled': self.net_metering_enabled
            }

            # Generate strategy using LLM
            strategy_prompt = f"""
            You are a solar battery optimization expert. Determine the optimal energy storage strategy.

            Current System State:
            - Net energy flow: {net_energy:.1f} kW (positive = excess solar, negative = deficit)
            - Battery state of charge: {battery_soc:.1f}%
            - Energy price tier: {price_tier}
            - Current price: ${current_price:.4f}/kWh
            - Backup reserve required: {self.backup_power_reserved}%

            System Configuration:
            - Storage strategy: {self.storage_strategy}
            - Net metering enabled: {self.net_metering_enabled}

            6-Hour Generation Forecast:
            {json.dumps(self.generation_forecast[:6], indent=2)}

            Determine the optimal action considering:
            1. Self-consumption maximization
            2. Battery state of charge management
            3. Grid arbitrage opportunities
            4. Backup power requirements
            5. Peak shaving benefits

            Respond in JSON format:
            {{
                "primary_action": "<charge_battery|discharge_battery|export_to_grid|import_from_grid|maintain>",
                "target_power_kw": <power_level>,
                "duration_minutes": <expected_duration>,
                "reasoning": "<explanation>",
                "confidence": <0-1>,
                "secondary_actions": ["<additional_actions>"]
            }}
            """

            response = await self.llm_client.generate_response(
                prompt=strategy_prompt,
                temperature=0.3,
                max_tokens=600
            )

            strategy = self.llm_client._parse_json_response(response)

            if 'primary_action' in strategy:
                return strategy
            else:
                logger.warning("Failed to generate valid storage strategy")
                return self._get_default_strategy(net_energy, battery_soc)

        except Exception as e:
            logger.error(f"Error determining optimal strategy: {e}")
            return self._get_default_strategy(net_energy, battery_soc)

    def _get_default_strategy(self, net_energy: float, battery_soc: float) -> \
    Dict[str, Any]:
        """Get default storage strategy when LLM fails"""
        if net_energy > 1.0 and battery_soc < 90:  # Excess solar, battery not full
            return {
                'primary_action': 'charge_battery',
                'target_power_kw': min(net_energy, 5.0),  # Max 5kW charging
                'reasoning': 'Default: charge battery with excess solar'
            }
        elif net_energy < -1.0 and battery_soc > self.backup_power_reserved + 10:  # Energy deficit, battery available
            return {
                'primary_action': 'discharge_battery',
                'target_power_kw': min(abs(net_energy), 5.0),
                # Max 5kW discharging
                'reasoning': 'Default: discharge battery to meet demand'
            }
        else:
            return {
                'primary_action': 'maintain',
                'target_power_kw': 0,
                'reasoning': 'Default: maintain current state'
            }

    async def _execute_storage_strategy(self, strategy: Dict[str, Any]):
        """Execute the determined storage strategy"""
        try:
            action = strategy.get('primary_action', 'maintain')
            target_power = strategy.get('target_power_kw', 0)

            if action == 'charge_battery':
                await self._charge_batteries(target_power)
            elif action == 'discharge_battery':
                await self._discharge_batteries(target_power)
            elif action == 'export_to_grid':
                await self._export_to_grid(target_power)
            elif action == 'import_from_grid':
                await self._import_from_grid(target_power)
            # 'maintain' requires no action

            # Log strategy execution
            logger.info(
                f"Executed storage strategy: {action} ({target_power:.1f}kW) - {strategy.get('reasoning', '')}")

        except Exception as e:
            logger.error(f"Error executing storage strategy: {e}")

    async def _charge_batteries(self, target_power_kw: float):
        """Charge batteries with specified power"""
        try:
            total_available_power = 0

            # Calculate total available charging power
            for device_data in self.battery_devices.values():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)
                max_power = device_data['max_power_kw']

                if soc < 95:  # Don't overcharge
                    total_available_power += max_power

            # Distribute charging power across batteries
            actual_power = min(target_power_kw, total_available_power)

            if actual_power > 0:
                power_per_battery = actual_power / len(self.battery_devices)

                for device_id, device_data in self.battery_devices.items():
                    current_state = device_data.get('current_state', {})
                    soc = current_state.get('state_of_charge', 50)

                    if soc < 95:
                        # Set charging power (negative power flow = charging)
                        charging_power = -min(power_per_battery, device_data[
                            'max_power_kw']) * 1000  # Convert to W

                        success = await self.device_service.set_device_state(
                            device_id,
                            {'power_flow': charging_power}
                        )

                        if success:
                            logger.debug(
                                f"Set battery {device_id} to charge at {abs(charging_power) / 1000:.1f}kW")

        except Exception as e:
            logger.error(f"Error charging batteries: {e}")

    async def _discharge_batteries(self, target_power_kw: float):
        """Discharge batteries with specified power"""
        try:
            total_available_power = 0

            # Calculate total available discharging power
            for device_data in self.battery_devices.values():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)
                max_power = device_data['max_power_kw']
                backup_reserve = self.backup_power_reserved

                if soc > backup_reserve + 5:  # Keep backup reserve + 5% buffer
                    total_available_power += max_power

            # Distribute discharging power across batteries
            actual_power = min(target_power_kw, total_available_power)

            if actual_power > 0:
                power_per_battery = actual_power / len(self.battery_devices)

                for device_id, device_data in self.battery_devices.items():
                    current_state = device_data.get('current_state', {})
                    soc = current_state.get('state_of_charge', 50)
                    backup_reserve = self.backup_power_reserved

                    if soc > backup_reserve + 5:
                        # Set discharging power (positive power flow = discharging)
                        discharging_power = min(power_per_battery, device_data[
                            'max_power_kw']) * 1000  # Convert to W

                        success = await self.device_service.set_device_state(
                            device_id,
                            {'power_flow': discharging_power}
                        )

                        if success:
                            logger.debug(
                                f"Set battery {device_id} to discharge at {discharging_power / 1000:.1f}kW")

        except Exception as e:
            logger.error(f"Error discharging batteries: {e}")

    async def _export_to_grid(self, target_power_kw: float):
        """Export power to grid (net metering)"""
        try:
            if not self.net_metering_enabled:
                logger.info("Net metering not enabled, cannot export to grid")
                return

            # In a real system, this would control grid-tie inverter
            logger.info(f"Exporting {target_power_kw:.1f}kW to grid")

            # Notify other agents about grid export
            await self.broadcast_message(
                'notification',
                {
                    'notification_type': 'grid_export_started',
                    'export_power_kw': target_power_kw,
                    'estimated_revenue': target_power_kw * 0.10
                    # $0.10/kWh export rate
                }
            )

        except Exception as e:
            logger.error(f"Error exporting to grid: {e}")

    async def _import_from_grid(self, target_power_kw: float):
        """Import power from grid"""
        try:
            # This would be handled by the grid-tie system automatically
            logger.info(f"Importing {target_power_kw:.1f}kW from grid")

        except Exception as e:
            logger.error(f"Error importing from grid: {e}")

    async def _manage_grid_interaction(self):
        """Manage interaction with the electrical grid"""
        try:
            # Calculate net power flow
            total_generation = sum(
                device.get('current_state', {}).get('power_generation', 0)
                for device in self.solar_devices.values()
            ) / 1000  # kW

            total_battery_flow = sum(
                device.get('current_state', {}).get('power_flow', 0)
                for device in self.battery_devices.values()
            ) / 1000  # kW (positive = discharging, negative = charging)

            # Get current home consumption
            try:
                consumption_response = await self._request_from_agent(
                    'energy_monitor', 'get_consumption_data', {'hours': 1}
                )
                recent_consumption = consumption_response.get(
                    'consumption_data', [])
                current_consumption = sum(
                    data.get('energy_consumption', 0) for data in
                    recent_consumption[-1:]
                ) if recent_consumption else 3.0
            except:
                current_consumption = 3.0

            # Calculate net grid flow
            net_grid_flow = total_generation + total_battery_flow - current_consumption

            # Store grid interaction data
            grid_data = {
                'timestamp': datetime.utcnow(),
                'solar_generation_kw': total_generation,
                'battery_flow_kw': total_battery_flow,
                'home_consumption_kw': current_consumption,
                'net_grid_flow_kw': net_grid_flow,
                'grid_export': net_grid_flow > 0,
                'self_consumption_ratio': min(1.0,
                                              current_consumption / max(0.1,
                                                                        total_generation))
            }

            await self.db_service.store_document('energy_patterns', {
                'timestamp': datetime.utcnow(),
                'device_id': 'grid_interaction',
                'device_type': 'grid_meter',
                'grid_data': grid_data
            })

            # Check for grid export limits
            if net_grid_flow > 10:  # More than 10kW export
                logger.warning(
                    f"High grid export detected: {net_grid_flow:.1f}kW")

                # Reduce export by charging batteries more aggressively
                excess_power = net_grid_flow - 8  # Keep export under 8kW
                await self._charge_batteries(excess_power)

        except Exception as e:
            logger.error(f"Error managing grid interaction: {e}")

    async def _manage_backup_power(self):
        """Manage backup power capabilities"""
        try:
            for device_id, device_data in self.battery_devices.items():
                if not device_data.get('backup_capable', False):
                    continue

                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)

                # Check if backup power reserve is maintained
                if soc < self.backup_power_reserved:
                    # Priority charge for backup power
                    logger.warning(
                        f"Battery {device_id} below backup reserve ({soc:.1f}% < {self.backup_power_reserved}%)")

                    # Request priority charging
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'backup_power_low',
                            'device_id': device_id,
                            'current_soc': soc,
                            'required_soc': self.backup_power_reserved,
                            'severity': 'high' if soc < self.backup_power_reserved - 10 else 'medium'
                        }
                    )

        except Exception as e:
            logger.error(f"Error managing backup power: {e}")

    async def _detect_system_anomalies(self):
        """Detect anomalies in solar and battery systems"""
        try:
            current_time = datetime.utcnow()
            hour = current_time.hour

            # Check solar system anomalies
            for device_id, device_data in self.solar_devices.items():
                current_state = device_data.get('current_state', {})
                generation = current_state.get('power_generation',
                                               0) / 1000  # kW
                irradiance = current_state.get('irradiance', 0)  # W/m²

                # Check for underperformance during daylight hours
                if 8 <= hour <= 16 and irradiance > 500:  # Good solar conditions
                    expected_generation = device_data['capacity_kw'] * (
                                irradiance / 1000) * 0.8

                    if generation < expected_generation * 0.6:  # Less than 60% of expected
                        await self.broadcast_message(
                            'notification',
                            {
                                'notification_type': 'solar_underperformance',
                                'device_id': device_id,
                                'current_generation': generation,
                                'expected_generation': expected_generation,
                                'performance_ratio': generation / expected_generation if expected_generation > 0 else 0,
                                'severity': 'medium'
                            }
                        )

                        logger.warning(
                            f"Solar underperformance detected: {device_id} generating {generation:.1f}kW (expected {expected_generation:.1f}kW)")

            # Check battery system anomalies
            for device_id, device_data in self.battery_devices.items():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)
                power_flow = current_state.get('power_flow', 0) / 1000  # kW

                # Check for rapid SOC changes (potential fault)
                generation_history = device_data.get('generation_history', [])
                if len(generation_history) >= 2:
                    # Compare with previous reading (would need to track battery SOC history)
                    pass  # Simplified for this example

                # Check for stuck at extreme SOC
                if soc <= 5:
                    logger.error(
                        f"Battery {device_id} critically low: {soc:.1f}%")
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'battery_critical_low',
                            'device_id': device_id,
                            'current_soc': soc,
                            'severity': 'critical'
                        }
                    )
                elif soc >= 98 and abs(
                        power_flow) < 0.1:  # Full and not being used
                    logger.info(f"Battery {device_id} fully charged and idle")

        except Exception as e:
            logger.error(f"Error detecting system anomalies: {e}")

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

    async def _handle_generation_forecast_request(self, sender: str,
                                                  data: Dict[str, Any]):
        """Handle request for generation forecast"""
        hours = data.get('hours', 24)

        forecast_data = self.generation_forecast[
                        :hours] if self.generation_forecast else []

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'generation_forecast': forecast_data,
            'total_expected_generation': sum(
                f.get('generation_kw', 0) for f in forecast_data),
            'hours_forecasted': len(forecast_data)
        })

    async def _handle_battery_status_request(self, sender: str,
                                             data: Dict[str, Any]):
        """Handle request for battery status"""
        battery_status = {}

        for device_id, device_data in self.battery_devices.items():
            current_state = device_data.get('current_state', {})

            battery_status[device_id] = {
                'state_of_charge': current_state.get('state_of_charge', 50),
                'power_flow': current_state.get('power_flow', 0),
                'available_capacity': current_state.get('available_capacity',
                                                        0),
                'max_power_kw': device_data['max_power_kw'],
                'backup_capable': device_data.get('backup_capable', False)
            }

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'battery_status': battery_status,
            'backup_power_reserved': self.backup_power_reserved
        })

    async def _handle_weather_update(self, sender: str, data: Dict[str, Any]):
        """Handle weather update notifications"""
        logger.debug(
            "Weather update received, will update generation forecast")
        # Weather updates will trigger forecast update in next execution cycle

    async def _handle_price_update(self, sender: str, data: Dict[str, Any]):
        """Handle price update notifications"""
        price_tier = data.get('price_tier', 'standard')
        current_price = data.get('current_price', 0.12)

        logger.debug(f"Price update: {price_tier} @ ${current_price}/kWh")

        # Adjust storage strategy based on pricing
        if price_tier == 'peak' and current_price > 0.20:
            # High prices - prioritize discharging battery
            for device_id, device_data in self.battery_devices.items():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)

                if soc > self.backup_power_reserved + 15:  # Have excess capacity
                    # Discharge to support home load during peak pricing
                    await self.device_service.set_device_state(
                        device_id,
                        {'power_flow': 3000}  # 3kW discharge
                    )

                    logger.info(
                        f"Peak pricing: discharging battery {device_id}")

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Maximize battery discharge to support critical loads
            for device_id, device_data in self.battery_devices.items():
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)

                if soc > self.backup_power_reserved:
                    max_discharge = device_data[
                                        'max_power_kw'] * 1000  # Convert to W

                    await self.device_service.set_device_state(
                        device_id,
                        {'power_flow': max_discharge}
                    )

                    logger.warning(
                        f"Energy emergency: maximum discharge from battery {device_id}")

    async def _handle_grid_outage(self, sender: str, data: Dict[str, Any]):
        """Handle grid outage events"""
        outage_duration = data.get('estimated_duration_minutes', 60)

        logger.critical(
            f"Grid outage detected, estimated duration: {outage_duration} minutes")

        # Switch to backup power mode
        for device_id, device_data in self.battery_devices.items():
            if device_data.get('backup_capable', False):
                current_state = device_data.get('current_state', {})
                soc = current_state.get('state_of_charge', 50)
                capacity_kwh = device_data['capacity_kwh']

                # Calculate backup power duration
                available_energy = (soc / 100) * capacity_kwh
                backup_duration_hours = available_energy / 3.0  # Assume 3kW critical load

                logger.info(
                    f"Battery {device_id} backup duration: {backup_duration_hours:.1f} hours")

                # Set to backup mode (implementation would depend on inverter capabilities)
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'backup_power_activated',
                        'device_id': device_id,
                        'available_energy_kwh': available_energy,
                        'estimated_duration_hours': backup_duration_hours
                    }
                )

    async def _handle_high_consumption(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle high consumption notifications"""
        consumption_kw = data.get('total_consumption', 0)

        if consumption_kw > 8:  # High consumption threshold
            # Check if we should discharge battery to reduce grid load
            total_battery_capacity = sum(
                device_data.get('current_state', {}).get('state_of_charge',
                                                         50) *
                device_data['capacity_kwh'] / 100
                for device_data in self.battery_devices.values()
            )

            if total_battery_capacity > 5:  # Have at least 5kWh available
                # Discharge batteries to offset high consumption
                discharge_power = min(consumption_kw * 0.3,
                                      5.0)  # Up to 30% or 5kW

                await self._discharge_batteries(discharge_power)

                logger.info(
                    f"High consumption detected ({consumption_kw:.1f}kW), discharging {discharge_power:.1f}kW from batteries")