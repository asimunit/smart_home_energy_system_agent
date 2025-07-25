import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from core.base_agent import BaseAgent
from services.device_service import DeviceService
from config.settings import settings

logger = logging.getLogger(__name__)


class ApplianceAgent(BaseAgent):
    """Agent responsible for managing smart appliances for optimal energy efficiency"""

    def __init__(self):
        super().__init__("appliance", "Appliance")
        self.device_service = DeviceService()
        self.appliances = {}
        self.scheduled_operations = {}
        self.energy_optimization_mode = 'balanced'
        self.current_price_tier = 'standard'
        self.delay_tolerances = {
            'dishwasher': 4,  # 4 hours delay tolerance
            'washing_machine': 6,  # 6 hours delay tolerance
            'dryer': 2,  # 2 hours delay tolerance
            'oven': 0  # No delay tolerance for safety
        }

    async def initialize(self):
        """Initialize the appliance agent"""
        logger.info("Initializing Appliance Agent")

        # Load appliance devices
        await self._load_appliances()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'price_update', 'energy_emergency', 'demand_response_event',
                'optimal_schedule_available', 'energy_prediction_updated'
            ]
        )

        # Initialize scheduling system
        await self._initialize_scheduling()

        logger.info("Appliance Agent initialized successfully")

    async def execute(self):
        """Main execution logic for appliance management"""
        try:
            # Monitor appliance states
            await self._monitor_appliance_states()

            # Process scheduled operations
            await self._process_scheduled_operations()

            # Optimize current operations
            await self._optimize_operations()

            # Check for energy saving opportunities
            await self._identify_savings_opportunities()

            # Update schedules based on pricing
            await self._update_schedules_for_pricing()

        except Exception as e:
            logger.error(f"Error in appliance execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Appliance Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents or users"""
        request_type = data.get('request_type')

        if request_type == 'start_appliance':
            await self._handle_start_appliance_request(sender, data)
        elif request_type == 'schedule_operation':
            await self._handle_schedule_operation_request(sender, data)
        elif request_type == 'get_appliance_status':
            await self._handle_status_request(sender, data)
        elif request_type == 'cancel_operation':
            await self._handle_cancel_operation_request(sender, data)
        elif request_type == 'optimize_schedule':
            await self._handle_optimize_schedule_request(sender, data)
        elif request_type == 'emergency_stop':
            await self._handle_emergency_stop_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'price_update':
            await self._handle_price_update(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)
        elif notification_type == 'demand_response_event':
            await self._handle_demand_response(sender, data)
        elif notification_type == 'energy_prediction_updated':
            await self._handle_energy_prediction(sender, data)
        elif notification_type == 'pricing_recommendations':
            await self._handle_pricing_recommendations(sender, data)

    async def _load_appliances(self):
        """Load appliance devices from device service"""
        try:
            devices = await self.device_service.get_devices_by_type(
                'appliance')

            for device in devices:
                device_id = device['device_id']
                self.appliances[device_id] = {
                    'device_info': device,
                    'current_state': None,
                    'scheduled_operations': [],
                    'energy_consumption_pattern': {},
                    'last_optimization': None,
                    'delay_tolerance': self.delay_tolerances.get(
                        device.get('device_subtype', ''), 2
                    ),
                    'priority': self._get_appliance_priority(device),
                    'energy_efficiency_rating': device.get(
                        'energy_star_rating', 'B')
                }

            logger.info(f"Loaded {len(self.appliances)} appliances")

        except Exception as e:
            logger.error(f"Error loading appliances: {e}")

    def _get_appliance_priority(self, device: Dict[str, Any]) -> str:
        """Get priority level for appliance"""
        device_subtype = device.get('device_subtype', '')

        priority_map = {
            'refrigerator': 'critical',  # Must always run
            'water_heater': 'high',  # Important for comfort
            'dishwasher': 'medium',  # Can be delayed
            'washing_machine': 'medium',  # Can be delayed
            'dryer': 'medium',  # Can be delayed
            'oven': 'high',  # When in use, cannot delay
            'microwave': 'high'  # Quick use, no delay
        }

        return priority_map.get(device_subtype, 'medium')

    async def _initialize_scheduling(self):
        """Initialize the scheduling system"""
        try:
            # Load existing schedules from database
            for appliance_id in self.appliances:
                # Get recent scheduled operations
                query = {
                    "bool": {
                        "must": [
                            {"term": {"device_id": appliance_id}},
                            {"range": {"timestamp": {"gte": "now-1d"}}}
                        ]
                    }
                }

                scheduled_ops = await self.db_service.search_documents(
                    "device_states", query, size=50
                )

                # Process scheduled operations
                for op in scheduled_ops:
                    if 'scheduled_operations' in op:
                        self.appliances[appliance_id][
                            'scheduled_operations'].extend(
                            op['scheduled_operations']
                        )

            logger.info("Scheduling system initialized")

        except Exception as e:
            logger.error(f"Error initializing scheduling: {e}")

    async def _monitor_appliance_states(self):
        """Monitor current states of all appliances"""
        try:
            for appliance_id, appliance_data in self.appliances.items():
                # Get current device state
                current_state = await self.device_service.get_device_state(
                    appliance_id)

                if current_state:
                    appliance_data['current_state'] = current_state

                    # Analyze consumption pattern
                    consumption = current_state.get('power_consumption', 0)

                    # Store consumption data with embedding
                    consumption_data = {
                        'timestamp': datetime.utcnow(),
                        'device_id': appliance_id,
                        'energy_consumption': consumption / 1000,
                        # Convert W to kW
                        'device_type': 'appliance',
                        'device_subtype': appliance_data['device_info'].get(
                            'device_subtype'),
                        'cycle_state': current_state.get('cycle_state',
                                                         'idle'),
                        'cycle_type': current_state.get('cycle_type')
                    }

                    # Generate embedding for consumption pattern
                    embedding = await self.embedding_service.embed_energy_pattern(
                        consumption_data)
                    consumption_data['pattern_embedding'] = embedding

                    await self.db_service.store_energy_pattern(
                        consumption_data)

                    # Update consumption pattern analysis
                    await self._update_consumption_pattern(appliance_id,
                                                           consumption_data)

        except Exception as e:
            logger.error(f"Error monitoring appliance states: {e}")

    async def _update_consumption_pattern(self, appliance_id: str,
                                          consumption_data: Dict[str, Any]):
        """Update consumption pattern analysis for an appliance"""
        try:
            appliance_data = self.appliances[appliance_id]
            pattern = appliance_data['energy_consumption_pattern']

            cycle_state = consumption_data.get('cycle_state', 'idle')
            consumption = consumption_data.get('energy_consumption', 0)

            # Update pattern statistics
            if cycle_state not in pattern:
                pattern[cycle_state] = {
                    'total_consumption': 0,
                    'sample_count': 0,
                    'average_consumption': 0,
                    'peak_consumption': 0
                }

            pattern[cycle_state]['total_consumption'] += consumption
            pattern[cycle_state]['sample_count'] += 1
            pattern[cycle_state]['average_consumption'] = (
                    pattern[cycle_state]['total_consumption'] /
                    pattern[cycle_state]['sample_count']
            )
            pattern[cycle_state]['peak_consumption'] = max(
                pattern[cycle_state]['peak_consumption'],
                consumption
            )

        except Exception as e:
            logger.error(
                f"Error updating consumption pattern for {appliance_id}: {e}")

    async def _process_scheduled_operations(self):
        """Process scheduled operations that are due"""
        try:
            current_time = datetime.utcnow()

            for appliance_id, appliance_data in self.appliances.items():
                scheduled_ops = appliance_data['scheduled_operations']

                # Process due operations
                for operation in scheduled_ops[
                                 :]:  # Copy to avoid modification during iteration
                    scheduled_time = datetime.fromisoformat(
                        operation.get('start_time', ''))

                    if current_time >= scheduled_time:
                        # Execute the operation
                        success = await self._execute_scheduled_operation(
                            appliance_id, operation)

                        if success:
                            scheduled_ops.remove(operation)
                            logger.info(
                                f"Executed scheduled operation for {appliance_id}")
                        else:
                            # Retry later or handle failure
                            operation['retry_count'] = operation.get(
                                'retry_count', 0) + 1
                            if operation['retry_count'] > 3:
                                scheduled_ops.remove(operation)
                                logger.error(
                                    f"Failed to execute operation for {appliance_id} after 3 retries")

        except Exception as e:
            logger.error(f"Error processing scheduled operations: {e}")

    async def _execute_scheduled_operation(self, appliance_id: str,
                                           operation: Dict[str, Any]) -> bool:
        """Execute a scheduled operation"""
        try:
            operation_type = operation.get('operation_type')
            operation_params = operation.get('parameters', {})

            if operation_type == 'start_cycle':
                cycle_type = operation_params.get('cycle_type', 'normal')
                success = await self.device_service.start_device_cycle(
                    appliance_id, cycle_type)

                if success:
                    # Notify about operation start
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'appliance_operation_started',
                            'appliance_id': appliance_id,
                            'operation_type': operation_type,
                            'cycle_type': cycle_type,
                            'estimated_duration': operation_params.get(
                                'duration', 60),
                            'power_consumption': operation_params.get(
                                'power_consumption', 0)
                        }
                    )

                return success

            elif operation_type == 'set_state':
                new_state = operation_params.get('state', {})
                return await self.device_service.set_device_state(appliance_id,
                                                                  new_state)

            else:
                logger.warning(f"Unknown operation type: {operation_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing scheduled operation: {e}")
            return False

    async def _optimize_operations(self):
        """Optimize current appliance operations"""
        try:
            # Get current pricing information
            try:
                price_response = await self._request_from_agent(
                    'price_intelligence', 'get_current_price', {}
                )
                current_price = price_response.get('current_price', {})
                price_tier = current_price.get('price_tier', 'standard')
            except:
                price_tier = 'standard'

            # Check for operations that can be optimized
            for appliance_id, appliance_data in self.appliances.items():
                current_state = appliance_data.get('current_state', {})
                cycle_state = current_state.get('cycle_state', 'idle')

                # If appliance is idle and has scheduled operations, check if timing can be optimized
                if cycle_state == 'idle' and appliance_data[
                    'scheduled_operations']:
                    await self._optimize_appliance_schedule(appliance_id,
                                                            price_tier)

        except Exception as e:
            logger.error(f"Error optimizing operations: {e}")

    async def _optimize_appliance_schedule(self, appliance_id: str,
                                           price_tier: str):
        """Optimize schedule for a specific appliance"""
        try:
            appliance_data = self.appliances[appliance_id]
            scheduled_ops = appliance_data['scheduled_operations']
            delay_tolerance = appliance_data['delay_tolerance']

            if not scheduled_ops or delay_tolerance == 0:
                return

            # Get optimal timing recommendation
            try:
                optimal_response = await self._request_from_agent(
                    'price_intelligence', 'get_optimal_schedule', {
                        'device_type': 'appliance',
                        'energy_requirement': self._estimate_cycle_energy(
                            appliance_id)
                    }
                )

                optimal_times = optimal_response.get('optimal_schedule', [])
            except:
                optimal_times = []

            if not optimal_times:
                return

            # Find the best time within delay tolerance
            current_time = datetime.utcnow()
            best_time = None
            best_savings = 0

            for time_slot in optimal_times:
                slot_time = datetime.utcnow().replace(hour=time_slot['hour'],
                                                      minute=0, second=0)

                # Check if within delay tolerance
                for operation in scheduled_ops:
                    original_time = datetime.fromisoformat(
                        operation.get('start_time', ''))
                    time_diff = (
                                            slot_time - original_time).total_seconds() / 3600  # Hours

                    if 0 <= time_diff <= delay_tolerance:
                        savings = time_slot.get('savings_estimate', 0)
                        if savings > best_savings:
                            best_time = slot_time
                            best_savings = savings

            # Reschedule if beneficial
            if best_time and best_savings > 10:  # At least 10% savings
                for operation in scheduled_ops:
                    original_time = datetime.fromisoformat(
                        operation.get('start_time', ''))
                    time_diff = (
                                            best_time - original_time).total_seconds() / 3600

                    if 0 <= time_diff <= delay_tolerance:
                        operation['start_time'] = best_time.isoformat()
                        operation[
                            'optimization_reason'] = f"Rescheduled for {best_savings:.1f}% energy savings"

                        logger.info(
                            f"Rescheduled {appliance_id} for {best_savings:.1f}% savings")

        except Exception as e:
            logger.error(f"Error optimizing schedule for {appliance_id}: {e}")

    def _estimate_cycle_energy(self, appliance_id: str) -> float:
        """Estimate energy consumption for a typical cycle"""
        appliance_data = self.appliances[appliance_id]
        device_info = appliance_data['device_info']
        power_rating = device_info.get('power_rating', 1000)  # Watts

        # Typical cycle durations (minutes)
        cycle_durations = {
            'dishwasher': 120,  # 2 hours
            'washing_machine': 45,  # 45 minutes
            'dryer': 60,  # 1 hour
            'oven': 30  # 30 minutes
        }

        device_subtype = device_info.get('device_subtype', '')
        duration_minutes = cycle_durations.get(device_subtype, 60)

        # Calculate energy in kWh
        energy_kwh = (power_rating * duration_minutes) / (
                    1000 * 60)  # Convert W*min to kWh

        return energy_kwh

    async def _identify_savings_opportunities(self):
        """Identify energy savings opportunities"""
        try:
            opportunities = []

            for appliance_id, appliance_data in self.appliances.items():
                device_info = appliance_data['device_info']
                current_state = appliance_data.get('current_state', {})
                pattern = appliance_data.get('energy_consumption_pattern', {})

                # Check for inefficient usage patterns
                if 'running' in pattern:
                    avg_consumption = pattern['running'].get(
                        'average_consumption', 0)
                    peak_consumption = pattern['running'].get(
                        'peak_consumption', 0)

                    # Check if consumption is higher than expected
                    expected_consumption = device_info.get('power_rating',
                                                           1000) / 1000  # kW

                    if avg_consumption > expected_consumption * 1.2:  # 20% higher than expected
                        opportunities.append({
                            'appliance_id': appliance_id,
                            'opportunity_type': 'high_consumption',
                            'description': f'Higher than expected energy consumption detected',
                            'potential_savings': '10-15%',
                            'recommendations': [
                                'Check appliance maintenance',
                                'Verify proper loading',
                                'Consider energy-efficient settings'
                            ]
                        })

                # Check for scheduling opportunities
                scheduled_ops = appliance_data.get('scheduled_operations', [])
                if scheduled_ops and appliance_data['delay_tolerance'] > 0:
                    opportunities.append({
                        'appliance_id': appliance_id,
                        'opportunity_type': 'scheduling_optimization',
                        'description': 'Operations can be rescheduled for better pricing',
                        'potential_savings': '15-25%',
                        'recommendations': [
                            'Enable automatic scheduling optimization',
                            'Increase delay tolerance if possible'
                        ]
                    })

            # Broadcast opportunities if found
            if opportunities:
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'energy_savings_opportunities',
                        'opportunities': opportunities,
                        'total_opportunities': len(opportunities)
                    }
                )

        except Exception as e:
            logger.error(f"Error identifying savings opportunities: {e}")

    async def _update_schedules_for_pricing(self):
        """Update schedules based on current pricing conditions"""
        try:
            if self.current_price_tier == 'peak':
                # During peak pricing, delay non-essential operations
                for appliance_id, appliance_data in self.appliances.items():
                    if appliance_data['priority'] in ['medium', 'low']:
                        await self._delay_non_essential_operations(
                            appliance_id)

            elif self.current_price_tier == 'off_peak':
                # During off-peak, accelerate delayed operations
                for appliance_id, appliance_data in self.appliances.items():
                    await self._accelerate_delayed_operations(appliance_id)

        except Exception as e:
            logger.error(f"Error updating schedules for pricing: {e}")

    async def _delay_non_essential_operations(self, appliance_id: str):
        """Delay non-essential operations during peak pricing"""
        try:
            appliance_data = self.appliances[appliance_id]
            scheduled_ops = appliance_data['scheduled_operations']
            delay_tolerance = appliance_data['delay_tolerance']

            current_time = datetime.utcnow()

            for operation in scheduled_ops:
                scheduled_time = datetime.fromisoformat(
                    operation.get('start_time', ''))

                # If operation is scheduled soon and can be delayed
                if (
                        scheduled_time - current_time).total_seconds() < 3600:  # Within 1 hour
                    if delay_tolerance > 0:
                        # Delay by 2-4 hours to avoid peak pricing
                        delay_hours = min(4, delay_tolerance)
                        new_time = scheduled_time + timedelta(
                            hours=delay_hours)

                        operation['start_time'] = new_time.isoformat()
                        operation['delay_reason'] = 'peak_pricing_avoidance'

                        logger.info(
                            f"Delayed {appliance_id} operation by {delay_hours} hours due to peak pricing")

        except Exception as e:
            logger.error(f"Error delaying operations for {appliance_id}: {e}")

    async def _accelerate_delayed_operations(self, appliance_id: str):
        """Accelerate delayed operations during off-peak pricing"""
        try:
            appliance_data = self.appliances[appliance_id]
            scheduled_ops = appliance_data['scheduled_operations']

            current_time = datetime.utcnow()

            for operation in scheduled_ops:
                scheduled_time = datetime.fromisoformat(
                    operation.get('start_time', ''))
                delay_reason = operation.get('delay_reason')

                # If operation was delayed due to pricing and we're in off-peak
                if delay_reason == 'peak_pricing_avoidance':
                    # Move operation to current time if reasonable
                    if (
                            scheduled_time - current_time).total_seconds() > 1800:  # More than 30 minutes away
                        operation['start_time'] = (current_time + timedelta(
                            minutes=15)).isoformat()
                        operation[
                            'acceleration_reason'] = 'off_peak_pricing_opportunity'

                        logger.info(
                            f"Accelerated {appliance_id} operation due to off-peak pricing")

        except Exception as e:
            logger.error(
                f"Error accelerating operations for {appliance_id}: {e}")

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

    async def _handle_start_appliance_request(self, sender: str,
                                              data: Dict[str, Any]):
        """Handle request to start an appliance"""
        appliance_id = data.get('appliance_id')
        cycle_type = data.get('cycle_type', 'normal')
        immediate = data.get('immediate', False)

        if appliance_id not in self.appliances:
            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'failed',
                'error': 'Appliance not found'
            })
            return

        if immediate:
            # Start immediately
            success = await self.device_service.start_device_cycle(
                appliance_id, cycle_type)
            status = 'success' if success else 'failed'
        else:
            # Schedule for optimal time
            optimal_time = await self._find_optimal_time(appliance_id)

            operation = {
                'operation_type': 'start_cycle',
                'parameters': {
                    'cycle_type': cycle_type,
                    'duration': self._get_cycle_duration(appliance_id,
                                                         cycle_type),
                    'power_consumption': self._estimate_cycle_energy(
                        appliance_id)
                },
                'start_time': optimal_time.isoformat(),
                'requester': sender
            }

            self.appliances[appliance_id]['scheduled_operations'].append(
                operation)
            status = 'scheduled'

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'status': status,
            'appliance_id': appliance_id,
            'cycle_type': cycle_type
        })

    async def _find_optimal_time(self, appliance_id: str) -> datetime:
        """Find optimal time to run appliance"""
        try:
            appliance_data = self.appliances[appliance_id]
            delay_tolerance = appliance_data['delay_tolerance']

            # Get price forecast
            try:
                forecast_response = await self._request_from_agent(
                    'price_intelligence', 'get_price_forecast',
                    {'hours': delay_tolerance}
                )

                price_forecast = forecast_response.get('price_forecast',
                                                       {}).get('predictions',
                                                               [])
            except:
                price_forecast = []

            if price_forecast:
                # Find hour with lowest predicted price
                best_hour = min(price_forecast,
                                key=lambda x: x.get('predicted_consumption',
                                                    float('inf')))
                optimal_time = datetime.utcnow().replace(
                    hour=best_hour['hour'],
                    minute=0,
                    second=0,
                    microsecond=0
                )

                # Ensure it's in the future
                if optimal_time <= datetime.utcnow():
                    optimal_time += timedelta(days=1)

                return optimal_time
            else:
                # Default to off-peak hours (late night)
                return datetime.utcnow().replace(hour=2, minute=0, second=0,
                                                 microsecond=0) + timedelta(
                    days=1)

        except Exception as e:
            logger.error(f"Error finding optimal time: {e}")
            return datetime.utcnow() + timedelta(hours=2)

    def _get_cycle_duration(self, appliance_id: str, cycle_type: str) -> int:
        """Get cycle duration in minutes"""
        appliance_data = self.appliances[appliance_id]
        device_subtype = appliance_data['device_info'].get('device_subtype',
                                                           '')

        durations = {
            'dishwasher': {'normal': 120, 'eco': 180, 'quick': 60},
            'washing_machine': {'normal': 45, 'delicate': 30, 'heavy_duty': 60,
                                'quick': 20},
            'dryer': {'normal': 60, 'delicate': 40, 'heavy_duty': 80,
                      'air_dry': 120}
        }

        return durations.get(device_subtype, {}).get(cycle_type, 60)

    async def _handle_price_update(self, sender: str, data: Dict[str, Any]):
        """Handle price update notifications"""
        new_price_tier = data.get('price_tier', 'standard')

        if new_price_tier != self.current_price_tier:
            logger.info(
                f"Price tier changed from {self.current_price_tier} to {new_price_tier}")
            self.current_price_tier = new_price_tier

            # Trigger schedule optimization
            await self._update_schedules_for_pricing()

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Stop all non-essential appliances
            for appliance_id, appliance_data in self.appliances.items():
                if appliance_data['priority'] in ['medium', 'low']:
                    current_state = appliance_data.get('current_state', {})
                    if current_state.get('cycle_state') == 'running':
                        # Attempt to pause or stop the appliance
                        await self.device_service.set_device_state(
                            appliance_id,
                            {'power_state': 'off'}
                        )

                        logger.warning(f"Emergency stop for {appliance_id}")

    async def _handle_demand_response(self, sender: str, data: Dict[str, Any]):
        """Handle demand response events"""
        event_type = data.get('event_type')
        target_reduction = data.get('recommended_reduction', 20)

        if event_type == 'peak_demand_reduction':
            # Delay or reduce appliance operations
            total_delayed_power = 0

            for appliance_id, appliance_data in self.appliances.items():
                if appliance_data['priority'] in ['medium', 'low']:
                    current_state = appliance_data.get('current_state', {})
                    power_consumption = current_state.get('power_consumption',
                                                          0)

                    if current_state.get('cycle_state') == 'running':
                        # Calculate if we should delay this appliance
                        if total_delayed_power < target_reduction * 10:  # Rough calculation
                            # Delay remaining operations
                            scheduled_ops = appliance_data[
                                'scheduled_operations']
                            for operation in scheduled_ops:
                                operation['start_time'] = (
                                        datetime.fromisoformat(
                                            data.get('end_time')) +
                                        timedelta(minutes=30)
                                ).isoformat()
                                operation['delay_reason'] = 'demand_response'

                            total_delayed_power += power_consumption

                            logger.info(
                                f"Delayed {appliance_id} for demand response")

    async def _handle_energy_prediction(self, sender: str,
                                        data: Dict[str, Any]):
        """Handle energy prediction updates"""
        prediction = data.get('prediction')
        peak_hours = data.get('peak_hours', [])

        if prediction and peak_hours:
            # Adjust schedules to avoid predicted peak hours
            current_time = datetime.utcnow()

            for appliance_id, appliance_data in self.appliances.items():
                scheduled_ops = appliance_data['scheduled_operations']

                for operation in scheduled_ops:
                    scheduled_time = datetime.fromisoformat(
                        operation.get('start_time', ''))
                    scheduled_hour = scheduled_time.hour

                    # If operation is scheduled during predicted peak hours
                    if scheduled_hour in peak_hours:
                        # Find alternative time
                        alternative_hours = [h for h in range(24) if
                                             h not in peak_hours]
                        if alternative_hours:
                            # Choose earliest available hour after current time
                            current_hour = current_time.hour
                            next_available = min(
                                [h for h in alternative_hours if
                                 h > current_hour],
                                default=alternative_hours[0])

                            new_time = scheduled_time.replace(
                                hour=next_available)
                            if new_time <= scheduled_time:  # If next day
                                new_time += timedelta(days=1)

                            operation['start_time'] = new_time.isoformat()
                            operation[
                                'optimization_reason'] = 'avoid_predicted_peak'

                            logger.info(
                                f"Rescheduled {appliance_id} to avoid predicted peak hours")

    async def _handle_pricing_recommendations(self, sender: str,
                                              data: Dict[str, Any]):
        """Handle pricing recommendations"""
        recommendations = data.get('recommendations', [])

        for rec in recommendations:
            if rec.get('type') == 'timing_optimization':
                optimal_time = rec.get('optimal_time')
                potential_savings = rec.get('potential_savings', 0)

                if optimal_time and potential_savings > 15:  # Significant savings
                    # Apply to applicable appliances
                    optimal_hour = int(optimal_time.split(':')[0])

                    for appliance_id, appliance_data in self.appliances.items():
                        if (appliance_data['delay_tolerance'] > 0 and
                                appliance_data['scheduled_operations']):

                            for operation in appliance_data[
                                'scheduled_operations']:
                                current_time = datetime.fromisoformat(
                                    operation.get('start_time', ''))
                                new_time = current_time.replace(
                                    hour=optimal_hour, minute=0)

                                if new_time <= current_time:
                                    new_time += timedelta(days=1)

                                # Check if within delay tolerance
                                delay_hours = (
                                                          new_time - current_time).total_seconds() / 3600
                                if 0 <= delay_hours <= appliance_data[
                                    'delay_tolerance']:
                                    operation[
                                        'start_time'] = new_time.isoformat()
                                    operation[
                                        'optimization_reason'] = f"pricing_recommendation_{potential_savings:.1f}%_savings"