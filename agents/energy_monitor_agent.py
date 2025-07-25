import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

from core.base_agent import BaseAgent
from services.device_service import DeviceService
from config.settings import settings

logger = logging.getLogger(__name__)


class EnergyMonitorAgent(BaseAgent):
    """Agent responsible for monitoring energy consumption patterns and detecting anomalies"""

    def __init__(self):
        super().__init__("energy_monitor", "EnergyMonitor")
        self.device_service = DeviceService()
        self.monitoring_devices = {}
        self.consumption_baselines = {}
        self.anomaly_threshold = 0.3  # 30% deviation
        self.pattern_analysis_window = 168  # 7 days in hours

    async def initialize(self):
        """Initialize the energy monitor agent"""
        logger.info("Initializing Energy Monitor Agent")

        # Load device configurations
        await self._load_device_configurations()

        # Subscribe to device events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            ['device_state_change', 'new_device_added', 'device_removed']
        )

        # Initialize consumption baselines
        await self._initialize_baselines()

        logger.info("Energy Monitor Agent initialized successfully")

    async def execute(self):
        """Main execution logic for energy monitoring"""
        try:
            # Monitor current energy consumption
            await self._monitor_real_time_consumption()

            # Analyze consumption patterns
            await self._analyze_consumption_patterns()

            # Detect anomalies
            await self._detect_anomalies()

            # Update consumption predictions
            await self._update_consumption_predictions()

            # Generate efficiency recommendations
            await self._generate_efficiency_recommendations()

        except Exception as e:
            logger.error(f"Error in energy monitor execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Energy Monitor Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents"""
        request_type = data.get('request_type')

        if request_type == 'get_consumption_data':
            await self._handle_consumption_data_request(sender, data)
        elif request_type == 'get_efficiency_analysis':
            await self._handle_efficiency_analysis_request(sender, data)
        elif request_type == 'get_anomaly_report':
            await self._handle_anomaly_report_request(sender, data)
        elif request_type == 'set_monitoring_parameters':
            await self._handle_monitoring_parameters_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'device_state_change':
            await self._handle_device_state_change(data)
        elif notification_type == 'price_change':
            await self._handle_price_change(data)
        elif notification_type == 'weather_update':
            await self._handle_weather_update(data)

    async def _load_device_configurations(self):
        """Load device configurations from database"""
        try:
            # Get all monitored devices
            devices = await self.device_service.get_all_devices()

            for device in devices:
                device_id = device['device_id']
                self.monitoring_devices[device_id] = {
                    'device_type': device['device_type'],
                    'room': device.get('room', 'unknown'),
                    'power_rating': device.get('power_rating', 0),
                    'monitoring_enabled': device.get('monitoring_enabled',
                                                     True),
                    'last_reading': None,
                    'consumption_history': []
                }

            logger.info(
                f"Loaded {len(self.monitoring_devices)} devices for monitoring")

        except Exception as e:
            logger.error(f"Error loading device configurations: {e}")

    async def _initialize_baselines(self):
        """Initialize consumption baselines for each device"""
        try:
            for device_id in self.monitoring_devices:
                # Get historical consumption data (last 30 days)
                historical_data = await self.db_service.get_recent_energy_data(
                    device_id=device_id,
                    hours=720  # 30 days
                )

                if historical_data:
                    consumptions = [data['energy_consumption'] for data in
                                    historical_data]

                    baseline = {
                        'mean_consumption': np.mean(consumptions),
                        'std_consumption': np.std(consumptions),
                        'median_consumption': np.median(consumptions),
                        'percentile_90': np.percentile(consumptions, 90),
                        'baseline_updated': datetime.utcnow()
                    }

                    self.consumption_baselines[device_id] = baseline
                    logger.debug(
                        f"Initialized baseline for device {device_id}")
                else:
                    logger.warning(
                        f"No historical data for device {device_id}")

        except Exception as e:
            logger.error(f"Error initializing baselines: {e}")

    async def _monitor_real_time_consumption(self):
        """Monitor real-time energy consumption"""
        try:
            for device_id, device_info in self.monitoring_devices.items():
                if not device_info['monitoring_enabled']:
                    continue

                # Get current consumption reading
                current_reading = await self.device_service.get_device_consumption(
                    device_id)

                if current_reading:
                    # Store consumption data
                    consumption_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'energy_consumption': current_reading[
                            'power_consumption'],
                        'device_type': device_info['device_type'],
                        'room': device_info['room']
                    }

                    # Generate embedding for pattern analysis
                    embedding = await self.embedding_service.embed_energy_pattern(
                        consumption_data)
                    consumption_data['pattern_embedding'] = embedding

                    # Store in database
                    await self.db_service.store_energy_pattern(
                        consumption_data)

                    # Update device history
                    device_info['last_reading'] = current_reading
                    device_info['consumption_history'].append(consumption_data)

                    # Keep only recent history (last 100 readings)
                    if len(device_info['consumption_history']) > 100:
                        device_info['consumption_history'] = device_info[
                                                                 'consumption_history'][
                                                             -100:]

        except Exception as e:
            logger.error(f"Error monitoring real-time consumption: {e}")

    async def _analyze_consumption_patterns(self):
        """Analyze energy consumption patterns"""
        try:
            for device_id in self.monitoring_devices:
                # Get recent consumption data for pattern analysis
                recent_data = await self.db_service.get_recent_energy_data(
                    device_id=device_id,
                    hours=self.pattern_analysis_window
                )

                if not recent_data:
                    continue

                # Analyze patterns using LLM
                pattern_context = {
                    'device_id': device_id,
                    'device_type': self.monitoring_devices[device_id][
                        'device_type'],
                    'room': self.monitoring_devices[device_id]['room'],
                    'analysis_period': f"{self.pattern_analysis_window} hours",
                    'data_points': len(recent_data),
                    'consumption_data': recent_data[-24:]
                    # Last 24 hours for context
                }

                analysis = await self.llm_client.analyze_energy_pattern(
                    recent_data[-1],  # Latest data point
                    pattern_context
                )

                if analysis and 'efficiency_score' in analysis:
                    # Store pattern analysis
                    pattern_record = {
                        'timestamp': datetime.utcnow(),
                        'device_id': device_id,
                        'pattern_analysis': analysis,
                        'efficiency_score': analysis['efficiency_score'],
                        'recommendations': analysis.get('recommendations', [])
                    }

                    # Generate embedding for pattern
                    pattern_text = f"Device: {device_id}, Efficiency: {analysis['efficiency_score']}, Pattern: {analysis.get('pattern_type', 'unknown')}"
                    embedding = await self.embedding_service.generate_embedding(
                        pattern_text)
                    pattern_record['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(pattern_record)

                    # Notify other agents if efficiency is low
                    if analysis['efficiency_score'] < 50:
                        await self.broadcast_message(
                            'notification',
                            {
                                'notification_type': 'low_efficiency_detected',
                                'device_id': device_id,
                                'efficiency_score': analysis[
                                    'efficiency_score'],
                                'recommendations': analysis.get(
                                    'recommendations', [])
                            }
                        )

        except Exception as e:
            logger.error(f"Error analyzing consumption patterns: {e}")

    async def _detect_anomalies(self):
        """Detect consumption anomalies"""
        try:
            for device_id, baseline in self.consumption_baselines.items():
                device_info = self.monitoring_devices.get(device_id)
                if not device_info or not device_info['last_reading']:
                    continue

                current_consumption = device_info['last_reading'][
                    'power_consumption']
                mean_consumption = baseline['mean_consumption']
                std_consumption = baseline['std_consumption']

                # Calculate z-score
                if std_consumption > 0:
                    z_score = abs(
                        current_consumption - mean_consumption) / std_consumption

                    # Check for anomaly
                    if z_score > 2.0:  # 2 standard deviations
                        anomaly_severity = 'high' if z_score > 3.0 else 'medium'

                        anomaly_data = {
                            'timestamp': datetime.utcnow(),
                            'device_id': device_id,
                            'anomaly_type': 'consumption_deviation',
                            'severity': anomaly_severity,
                            'current_consumption': current_consumption,
                            'expected_consumption': mean_consumption,
                            'deviation_percentage': ((
                                                                 current_consumption - mean_consumption) / mean_consumption) * 100,
                            'z_score': z_score
                        }

                        # Store anomaly
                        await self.db_service.store_document('energy_patterns',
                                                             {
                                                                 **anomaly_data,
                                                                 'pattern_type': 'anomaly'
                                                             })

                        # Notify other agents
                        await self.broadcast_message(
                            'notification',
                            {
                                'notification_type': 'energy_anomaly_detected',
                                **anomaly_data
                            }
                        )

                        logger.warning(
                            f"Anomaly detected for device {device_id}: {anomaly_severity} severity")

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

    async def _update_consumption_predictions(self):
        """Update consumption predictions"""
        try:
            # Get historical data for prediction
            all_recent_data = await self.db_service.get_recent_energy_data(
                hours=168)  # 7 days

            if not all_recent_data:
                return

            # Prepare data for prediction
            historical_context = {
                'total_devices': len(self.monitoring_devices),
                'analysis_period': '7 days',
                'data_points': len(all_recent_data),
                'consumption_trends': all_recent_data[-48:]  # Last 48 hours
            }

            # Get external factors
            external_factors = {
                'season': self._get_current_season(),
                'day_of_week': datetime.utcnow().strftime('%A'),
                'hour': datetime.utcnow().hour
            }

            # Generate prediction using LLM
            prediction = await self.llm_client.predict_energy_demand(
                historical_context,
                external_factors
            )

            if prediction and 'hourly_predictions' in prediction:
                # Store prediction
                prediction_record = {
                    'timestamp': datetime.utcnow(),
                    'prediction_type': 'energy_demand',
                    'prediction_data': prediction,
                    'confidence': prediction.get('confidence', 0.5)
                }

                await self.db_service.store_document('energy_patterns',
                                                     prediction_record)

                # Notify other agents about prediction
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'energy_prediction_updated',
                        'prediction': prediction,
                        'peak_hours': prediction.get('peak_hours', [])
                    }
                )

        except Exception as e:
            logger.error(f"Error updating consumption predictions: {e}")

    async def _generate_efficiency_recommendations(self):
        """Generate energy efficiency recommendations"""
        try:
            # Analyze overall system efficiency
            inefficient_devices = []

            for device_id, device_info in self.monitoring_devices.items():
                if device_info['consumption_history']:
                    recent_consumption = device_info['consumption_history'][
                                         -10:]  # Last 10 readings
                    avg_consumption = np.mean(
                        [r['energy_consumption'] for r in recent_consumption])

                    baseline = self.consumption_baselines.get(device_id)
                    if baseline and avg_consumption > baseline[
                        'percentile_90']:
                        inefficient_devices.append({
                            'device_id': device_id,
                            'device_type': device_info['device_type'],
                            'room': device_info['room'],
                            'avg_consumption': avg_consumption,
                            'baseline_90th': baseline['percentile_90'],
                            'efficiency_loss': ((avg_consumption - baseline[
                                'mean_consumption']) / baseline[
                                                    'mean_consumption']) * 100
                        })

            if inefficient_devices:
                # Generate recommendations
                recommendations = []
                for device in inefficient_devices:
                    recommendations.append({
                        'device_id': device['device_id'],
                        'recommendation_type': 'efficiency_improvement',
                        'message': f"Device {device['device_id']} ({device['device_type']}) in {device['room']} is consuming {device['efficiency_loss']:.1f}% more energy than baseline",
                        'priority': 'high' if device[
                                                  'efficiency_loss'] > 50 else 'medium',
                        'potential_savings': device['efficiency_loss']
                    })

                # Broadcast recommendations
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'efficiency_recommendations',
                        'recommendations': recommendations,
                        'total_inefficient_devices': len(inefficient_devices)
                    }
                )

        except Exception as e:
            logger.error(f"Error generating efficiency recommendations: {e}")

    def _get_current_season(self) -> str:
        """Get current season based on date"""
        month = datetime.utcnow().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    async def _handle_consumption_data_request(self, sender: str,
                                               data: Dict[str, Any]):
        """Handle requests for consumption data"""
        device_id = data.get('device_id')
        hours = data.get('hours', 24)

        if device_id:
            consumption_data = await self.db_service.get_recent_energy_data(
                device_id, hours)
        else:
            consumption_data = await self.db_service.get_recent_energy_data(
                hours=hours)

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'consumption_data': consumption_data,
            'data_points': len(consumption_data)
        })

    async def _handle_efficiency_analysis_request(self, sender: str,
                                                  data: Dict[str, Any]):
        """Handle requests for efficiency analysis"""
        device_id = data.get('device_id')

        if device_id and device_id in self.monitoring_devices:
            device_info = self.monitoring_devices[device_id]
            baseline = self.consumption_baselines.get(device_id)

            analysis = {
                'device_id': device_id,
                'device_type': device_info['device_type'],
                'monitoring_enabled': device_info['monitoring_enabled'],
                'last_reading': device_info['last_reading'],
                'baseline': baseline,
                'history_points': len(device_info['consumption_history'])
            }
        else:
            analysis = {'error': 'Device not found or not monitored'}

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'efficiency_analysis': analysis
        })

    async def _handle_device_state_change(self, data: Dict[str, Any]):
        """Handle device state change notifications"""
        device_id = data.get('device_id')
        new_state = data.get('new_state')

        if device_id in self.monitoring_devices:
            logger.info(f"Device {device_id} state changed: {new_state}")
            # Update monitoring configuration if needed
            if 'power_state' in new_state:
                self.monitoring_devices[device_id][
                    'last_state_change'] = datetime.utcnow()

    async def _handle_price_change(self, data: Dict[str, Any]):
        """Handle energy price change notifications"""
        new_price = data.get('price_per_kwh')
        price_tier = data.get('price_tier')

        logger.info(f"Energy price changed: ${new_price}/kWh ({price_tier})")

        # Adjust monitoring sensitivity based on price
        if price_tier == 'peak':
            self.anomaly_threshold = 0.2  # More sensitive during peak pricing
        else:
            self.anomaly_threshold = 0.3  # Normal sensitivity

    async def _handle_weather_update(self, data: Dict[str, Any]):
        """Handle weather update notifications"""
        temperature = data.get('temperature')
        conditions = data.get('conditions')

        logger.debug(f"Weather update: {temperature}°F, {conditions}")

        # Adjust baselines for weather-sensitive devices (HVAC, etc.)
        for device_id, device_info in self.monitoring_devices.items():
            if device_info['device_type'] in ['hvac', 'heat_pump',
                                              'air_conditioner']:
                # Weather affects HVAC consumption patterns
                pass  # Could implement weather-based baseline adjustments