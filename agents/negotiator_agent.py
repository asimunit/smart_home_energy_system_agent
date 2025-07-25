import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from core.base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class AgentRequest:
    """Represents a request from an agent"""
    request_id: str
    sender_agent: str
    request_type: str
    priority: Priority
    data: Dict[str, Any]
    timestamp: datetime
    deadline: Optional[datetime] = None
    dependencies: List[str] = None


@dataclass
class NegotiationContext:
    """Context for negotiation decisions"""
    conflicting_requests: List[AgentRequest]
    system_constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]
    current_energy_state: Dict[str, Any]
    cost_implications: Dict[str, Any]


class NegotiatorAgent(BaseAgent):
    """Central coordinator agent responsible for negotiating between conflicting agent requests"""

    def __init__(self):
        super().__init__("negotiator", "Negotiator")
        self.active_requests = {}
        self.pending_negotiations = []
        self.resolution_history = []
        self.agent_registry = {}
        self.system_constraints = {}
        self.negotiation_timeout = 60  # 60 seconds

    async def initialize(self):
        """Initialize the negotiator agent"""
        logger.info("Initializing Negotiator Agent")

        # Subscribe to all system events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'agent_request', 'system_constraint_change', 'emergency_event',
                'user_preference_change', 'energy_state_change'
            ]
        )

        # Load system constraints
        await self._load_system_constraints()

        # Register with all agents
        await self._register_with_agents()

        logger.info("Negotiator Agent initialized successfully")

    async def execute(self):
        """Main execution logic for negotiation"""
        try:
            # Process pending requests
            await self._process_pending_requests()

            # Handle active negotiations
            await self._handle_active_negotiations()

            # Monitor system health
            await self._monitor_system_health()

            # Update agent registry
            await self._update_agent_registry()

            # Clean up expired requests
            await self._cleanup_expired_requests()

        except Exception as e:
            logger.error(f"Error in negotiator execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Negotiator Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents"""
        request_type = data.get('request_type')

        if request_type == 'negotiate_conflict':
            await self._handle_negotiation_request(sender, data)
        elif request_type == 'get_system_status':
            await self._handle_system_status_request(sender, data)
        elif request_type == 'register_agent':
            await self._handle_agent_registration(sender, data)
        elif request_type == 'update_constraints':
            await self._handle_constraint_update(sender, data)
        elif request_type == 'emergency_override':
            await self._handle_emergency_override(sender, data)
        else:
            logger.warning(
                f"Unknown request type from {sender}: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'resource_conflict':
            await self._handle_resource_conflict(sender, data)
        elif notification_type == 'agent_status_change':
            await self._handle_agent_status_change(sender, data)
        elif notification_type == 'system_emergency':
            await self._handle_system_emergency(sender, data)
        elif notification_type == 'user_preference_override':
            await self._handle_user_override(sender, data)

    async def _load_system_constraints(self):
        """Load system-wide constraints"""
        try:
            self.system_constraints = {
                'max_total_power': 10000,  # 10kW max
                'min_battery_reserve': 0.2,  # 20% minimum battery
                'safety_temperature_limits': {'min': 60, 'max': 85},
                # Fahrenheit
                'peak_demand_limit': 8000,  # 8kW during peak hours
                'emergency_power_reserve': 1000,  # 1kW emergency reserve
                'device_priority_levels': {
                    'critical': ['security_system', 'medical_devices'],
                    'high': ['refrigerator', 'hvac'],
                    'medium': ['lighting', 'entertainment'],
                    'low': ['pool_pump', 'lawn_sprinkler']
                }
            }

            logger.info("System constraints loaded")

        except Exception as e:
            logger.error(f"Error loading system constraints: {e}")

    async def _register_with_agents(self):
        """Register this negotiator with all other agents"""
        try:
            # Known agent types in the system
            agent_types = [
                'energy_monitor', 'price_intelligence', 'hvac', 'appliance',
                'lighting', 'ev_charging', 'solar_battery',
                'comfort_optimization'
            ]

            for agent_type in agent_types:
                await self.send_message(
                    agent_type,
                    'notification',
                    {
                        'notification_type': 'negotiator_registration',
                        'negotiator_id': self.agent_id,
                        'capabilities': ['conflict_resolution',
                                         'resource_allocation',
                                         'system_coordination']
                    }
                )

            logger.info(f"Registered with {len(agent_types)} agent types")

        except Exception as e:
            logger.error(f"Error registering with agents: {e}")

    async def _process_pending_requests(self):
        """Process pending negotiation requests"""
        try:
            if not self.pending_negotiations:
                return

            for negotiation in self.pending_negotiations[
                               :]:  # Copy to avoid modification during iteration
                if await self._should_process_negotiation(negotiation):
                    await self._conduct_negotiation(negotiation)
                    self.pending_negotiations.remove(negotiation)
                elif self._is_negotiation_expired(negotiation):
                    await self._handle_expired_negotiation(negotiation)
                    self.pending_negotiations.remove(negotiation)

        except Exception as e:
            logger.error(f"Error processing pending requests: {e}")

    async def _handle_negotiation_request(self, sender: str,
                                          data: Dict[str, Any]):
        """Handle a negotiation request from an agent"""
        try:
            # Create agent request
            request = AgentRequest(
                request_id=data.get('request_id',
                                    f"req_{datetime.utcnow().timestamp()}"),
                sender_agent=sender,
                request_type=data.get('request_type'),
                priority=Priority(data.get('priority', 3)),
                data=data.get('request_data', {}),
                timestamp=datetime.utcnow(),
                deadline=datetime.fromisoformat(data['deadline']) if data.get(
                    'deadline') else None,
                dependencies=data.get('dependencies', [])
            )

            # Store active request
            self.active_requests[request.request_id] = request

            # Check for conflicts with existing requests
            conflicts = await self._detect_conflicts(request)

            if conflicts:
                # Create negotiation context
                negotiation_context = await self._create_negotiation_context(
                    [request] + conflicts)
                self.pending_negotiations.append(negotiation_context)

                logger.info(
                    f"Conflict detected for request {request.request_id}, starting negotiation")
            else:
                # No conflicts, approve directly
                await self._approve_request(request)

        except Exception as e:
            logger.error(f"Error handling negotiation request: {e}")

    async def _detect_conflicts(self, new_request: AgentRequest) -> List[
        AgentRequest]:
        """Detect conflicts between new request and existing requests"""
        conflicts = []

        try:
            for existing_request in self.active_requests.values():
                if existing_request.request_id == new_request.request_id:
                    continue

                # Check for resource conflicts
                if await self._has_resource_conflict(new_request,
                                                     existing_request):
                    conflicts.append(existing_request)

                # Check for timing conflicts
                if await self._has_timing_conflict(new_request,
                                                   existing_request):
                    conflicts.append(existing_request)

                # Check for constraint violations
                if await self._violates_constraints(new_request,
                                                    existing_request):
                    conflicts.append(existing_request)

            return list(set(conflicts))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return []

    async def _has_resource_conflict(self, req1: AgentRequest,
                                     req2: AgentRequest) -> bool:
        """Check if two requests conflict over resources"""
        try:
            req1_resources = set(req1.data.get('required_resources', []))
            req2_resources = set(req2.data.get('required_resources', []))

            # Check for overlapping resources
            if req1_resources.intersection(req2_resources):
                return True

            # Check for power consumption conflicts
            req1_power = req1.data.get('power_requirement', 0)
            req2_power = req2.data.get('power_requirement', 0)

            if req1_power + req2_power > self.system_constraints.get(
                    'max_total_power', 10000):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking resource conflict: {e}")
            return False

    async def _has_timing_conflict(self, req1: AgentRequest,
                                   req2: AgentRequest) -> bool:
        """Check if two requests have timing conflicts"""
        try:
            req1_start = req1.data.get('start_time')
            req1_end = req1.data.get('end_time')
            req2_start = req2.data.get('start_time')
            req2_end = req2.data.get('end_time')

            if not all([req1_start, req1_end, req2_start, req2_end]):
                return False

            # Convert to datetime objects
            req1_start = datetime.fromisoformat(req1_start)
            req1_end = datetime.fromisoformat(req1_end)
            req2_start = datetime.fromisoformat(req2_start)
            req2_end = datetime.fromisoformat(req2_end)

            # Check for overlapping time periods
            return (req1_start < req2_end and req2_start < req1_end)

        except Exception as e:
            logger.error(f"Error checking timing conflict: {e}")
            return False

    async def _violates_constraints(self, req1: AgentRequest,
                                    req2: AgentRequest) -> bool:
        """Check if combining two requests violates system constraints"""
        try:
            # Check total power constraint
            total_power = req1.data.get('power_requirement',
                                        0) + req2.data.get('power_requirement',
                                                           0)
            if total_power > self.system_constraints.get('max_total_power',
                                                         10000):
                return True

            # Check device priority conflicts
            req1_devices = req1.data.get('devices', [])
            req2_devices = req2.data.get('devices', [])

            for device in req1_devices:
                if device in req2_devices:
                    return True  # Same device cannot be controlled by multiple requests simultaneously

            return False

        except Exception as e:
            logger.error(f"Error checking constraint violations: {e}")
            return False

    async def _create_negotiation_context(self, conflicting_requests: List[
        AgentRequest]) -> NegotiationContext:
        """Create context for negotiation"""
        try:
            # Get current system state
            system_state = await self._get_current_system_state()

            # Get user preferences
            user_preferences = await self.db_service.get_user_preferences(
                'default_user')
            if not user_preferences:
                user_preferences = {'energy_priority': 'balanced',
                                    'comfort_level': 'medium'}

            # Calculate cost implications
            cost_implications = await self._calculate_cost_implications(
                conflicting_requests)

            return NegotiationContext(
                conflicting_requests=conflicting_requests,
                system_constraints=self.system_constraints,
                user_preferences=user_preferences,
                current_energy_state=system_state,
                cost_implications=cost_implications
            )

        except Exception as e:
            logger.error(f"Error creating negotiation context: {e}")
            return None

    async def _conduct_negotiation(self, context: NegotiationContext):
        """Conduct negotiation between conflicting requests"""
        try:
            if not context or not context.conflicting_requests:
                return

            # Prepare negotiation input for LLM
            negotiation_input = {
                'conflicting_requests': [
                    {
                        'agent': req.sender_agent,
                        'request_type': req.request_type,
                        'priority': req.priority.name,
                        'data': req.data,
                        'timestamp': req.timestamp.isoformat(),
                        'deadline': req.deadline.isoformat() if req.deadline else None
                    }
                    for req in context.conflicting_requests
                ],
                'system_state': context.current_energy_state,
                'user_preferences': context.user_preferences,
                'constraints': context.system_constraints,
                'cost_implications': context.cost_implications
            }

            # Get negotiation decision from LLM
            decision = await self.llm_client.negotiate_priorities(
                context.conflicting_requests,
                {
                    'system_state': context.current_energy_state,
                    'user_preferences': context.user_preferences,
                    'constraints': context.system_constraints
                }
            )

            if decision and 'resolution' in decision:
                await self._execute_negotiation_decision(context, decision)
            else:
                logger.error(
                    "Failed to get valid negotiation decision from LLM")
                await self._apply_default_resolution(context)

        except Exception as e:
            logger.error(f"Error conducting negotiation: {e}")
            await self._apply_default_resolution(context)

    async def _execute_negotiation_decision(self, context: NegotiationContext,
                                            decision: Dict[str, Any]):
        """Execute the negotiation decision"""
        try:
            resolution = decision['resolution']
            winning_agent = resolution.get('agent_id')
            modifications = resolution.get('modifications', [])

            # Apply modifications to requests
            for modification in modifications:
                agent_id = modification['agent_id']
                modified_request = modification['modified_request']
                reason = modification['reason']

                # Find the request and update it
                for request in context.conflicting_requests:
                    if request.sender_agent == agent_id:
                        # Send modification response to agent
                        await self.send_message(
                            agent_id,
                            'response',
                            {
                                'request_id': request.request_id,
                                'status': 'modified',
                                'modification': modified_request,
                                'reason': reason,
                                'confidence': decision.get('confidence', 0.5)
                            }
                        )
                        break

            # Approve winning request
            if winning_agent:
                for request in context.conflicting_requests:
                    if request.sender_agent == winning_agent:
                        await self._approve_request(request)
                        break

            # Store resolution history
            resolution_record = {
                'timestamp': datetime.utcnow(),
                'conflicting_requests': len(context.conflicting_requests),
                'decision': decision,
                'execution_status': 'completed'
            }

            self.resolution_history.append(resolution_record)

            # Store in database
            await self.store_decision(resolution_record)

            logger.info(
                f"Negotiation resolved: {winning_agent} approved with {len(modifications)} modifications")

        except Exception as e:
            logger.error(f"Error executing negotiation decision: {e}")

    async def _approve_request(self, request: AgentRequest):
        """Approve a request"""
        try:
            await self.send_message(
                request.sender_agent,
                'response',
                {
                    'request_id': request.request_id,
                    'status': 'approved',
                    'message': 'Request approved by negotiator',
                    'timestamp': datetime.utcnow().isoformat()
                }
            )

            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

            logger.info(
                f"Approved request {request.request_id} from {request.sender_agent}")

        except Exception as e:
            logger.error(f"Error approving request: {e}")

    async def _apply_default_resolution(self, context: NegotiationContext):
        """Apply default resolution when LLM negotiation fails"""
        try:
            if not context.conflicting_requests:
                return

            # Sort by priority and timestamp
            sorted_requests = sorted(
                context.conflicting_requests,
                key=lambda r: (r.priority.value, r.timestamp)
            )

            # Approve highest priority request
            winning_request = sorted_requests[0]
            await self._approve_request(winning_request)

            # Reject others
            for request in sorted_requests[1:]:
                await self.send_message(
                    request.sender_agent,
                    'response',
                    {
                        'request_id': request.request_id,
                        'status': 'rejected',
                        'reason': 'Resource conflict - lower priority',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )

            logger.info(
                f"Applied default resolution: approved {winning_request.sender_agent}")

        except Exception as e:
            logger.error(f"Error applying default resolution: {e}")

    async def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state for negotiation context"""
        try:
            # Get energy consumption data
            recent_consumption = await self.db_service.get_recent_energy_data(
                hours=1)
            total_consumption = sum(
                data.get('energy_consumption', 0) for data in
                recent_consumption)

            # Get current price data
            current_price = await self.db_service.get_current_price_data()

            system_state = {
                'total_power_consumption': total_consumption,
                'available_capacity': self.system_constraints.get(
                    'max_total_power', 10000) - total_consumption,
                'current_price_per_kwh': current_price.get('price_per_kwh',
                                                           0.12) if current_price else 0.12,
                'active_devices': len(recent_consumption),
                'system_health': 'healthy',
                # Could be enhanced with actual health checks
                'timestamp': datetime.utcnow().isoformat()
            }

            return system_state

        except Exception as e:
            logger.error(f"Error getting system state: {e}")
            return {}

    async def _calculate_cost_implications(self,
                                           requests: List[AgentRequest]) -> \
    Dict[str, Any]:
        """Calculate cost implications of different negotiation outcomes"""
        try:
            current_price = await self.db_service.get_current_price_data()
            price_per_kwh = current_price.get('price_per_kwh',
                                              0.12) if current_price else 0.12

            cost_implications = {}

            for request in requests:
                power_req = request.data.get('power_requirement', 0)
                duration = request.data.get('duration_hours', 1)

                cost = (
                                   power_req / 1000) * duration * price_per_kwh  # Convert W to kW

                cost_implications[request.sender_agent] = {
                    'estimated_cost': cost,
                    'power_requirement': power_req,
                    'duration_hours': duration,
                    'cost_per_hour': cost / duration if duration > 0 else 0
                }

            return cost_implications

        except Exception as e:
            logger.error(f"Error calculating cost implications: {e}")
            return {}

    async def _should_process_negotiation(self,
                                          context: NegotiationContext) -> bool:
        """Check if a negotiation should be processed now"""
        if not context or not context.conflicting_requests:
            return False

        # Check if any request has passed its deadline
        current_time = datetime.utcnow()
        for request in context.conflicting_requests:
            if request.deadline and current_time > request.deadline:
                return True

        # Check if all dependencies are resolved
        for request in context.conflicting_requests:
            if request.dependencies:
                for dep_id in request.dependencies:
                    if dep_id in self.active_requests:
                        return False  # Wait for dependency

        return True

    def _is_negotiation_expired(self, context: NegotiationContext) -> bool:
        """Check if a negotiation has expired"""
        if not context or not context.conflicting_requests:
            return True

        oldest_request = min(context.conflicting_requests,
                             key=lambda r: r.timestamp)
        return (
                    datetime.utcnow() - oldest_request.timestamp).seconds > self.negotiation_timeout

    async def _handle_expired_negotiation(self, context: NegotiationContext):
        """Handle expired negotiation"""
        logger.warning("Negotiation expired, applying default resolution")
        await self._apply_default_resolution(context)

    async def _monitor_system_health(self):
        """Monitor overall system health"""
        try:
            # Check agent health
            agent_count = len(self.agent_registry)
            active_agents = sum(1 for agent in self.agent_registry.values() if
                                agent.get('status') == 'active')

            # Check resource utilization
            system_state = await self._get_current_system_state()
            utilization = system_state.get('total_power_consumption',
                                           0) / self.system_constraints.get(
                'max_total_power', 10000)

            health_status = {
                'timestamp': datetime.utcnow(),
                'total_agents': agent_count,
                'active_agents': active_agents,
                'system_utilization': utilization,
                'active_negotiations': len(self.pending_negotiations),
                'active_requests': len(self.active_requests),
                'health_score': min(1.0,
                                    (active_agents / max(1, agent_count)) * (
                                                1 - utilization))
            }

            # Store health status
            await self.db_service.store_document('agent_decisions', {
                'agent_id': self.agent_id,
                'decision_type': 'health_monitor',
                'decision_data': health_status,
                'timestamp': datetime.utcnow()
            })

            # Alert if health is poor
            if health_status['health_score'] < 0.5:
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'system_health_warning',
                        'health_status': health_status,
                        'severity': 'high' if health_status[
                                                  'health_score'] < 0.3 else 'medium'
                    }
                )

        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")

    async def _update_agent_registry(self):
        """Update registry of known agents"""
        try:
            # This could be enhanced to ping agents and check their status
            current_time = datetime.utcnow()

            # Mark agents as inactive if they haven't been heard from in a while
            for agent_id, agent_info in self.agent_registry.items():
                last_seen = agent_info.get('last_seen')
                if last_seen and (
                        current_time - last_seen).seconds > 300:  # 5 minutes
                    agent_info['status'] = 'inactive'

        except Exception as e:
            logger.error(f"Error updating agent registry: {e}")

    async def _cleanup_expired_requests(self):
        """Clean up expired requests"""
        try:
            current_time = datetime.utcnow()
            expired_requests = []

            for request_id, request in self.active_requests.items():
                # Remove requests older than 1 hour
                if (current_time - request.timestamp).seconds > 3600:
                    expired_requests.append(request_id)

            for request_id in expired_requests:
                del self.active_requests[request_id]
                logger.debug(f"Cleaned up expired request: {request_id}")

        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")