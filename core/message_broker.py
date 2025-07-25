import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import redis.asyncio as redis
from config.settings import settings

logger = logging.getLogger(__name__)


class MessageBroker:
    """Message broker for inter-agent communication using Redis"""

    def __init__(self):
        self.redis_client = None
        self.pubsub = None
        self.subscriptions = {}
        self.message_queues = {}
        self._initialize_client()
        logger.info("Initialized message broker")

    def _initialize_client(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise

    async def send_message(self, recipient: str, message: Dict[str, Any]):
        """Send a message to a specific agent"""
        try:
            # Add metadata
            message['sent_at'] = datetime.utcnow().isoformat()
            message['broker_id'] = f"msg_{datetime.utcnow().timestamp()}"

            # Store message in recipient's queue
            queue_key = f"agent_queue:{recipient}"
            message_str = json.dumps(message)

            await self.redis_client.lpush(queue_key, message_str)

            # Set expiration for queue (24 hours)
            await self.redis_client.expire(queue_key, 86400)

            logger.debug(f"Sent message to {recipient}: {message['type']}")

        except Exception as e:
            logger.error(f"Error sending message to {recipient}: {e}")

    async def get_messages(self, agent_id: str, max_messages: int = 10) -> \
    List[Dict[str, Any]]:
        """Get messages for a specific agent"""
        try:
            queue_key = f"agent_queue:{agent_id}"
            messages = []

            # Get messages from queue (FIFO - using RPOP for oldest first)
            for _ in range(max_messages):
                message_str = await self.redis_client.rpop(queue_key)
                if not message_str:
                    break

                try:
                    message = json.loads(message_str)
                    messages.append(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding message: {e}")
                    continue

            return messages

        except Exception as e:
            logger.error(f"Error getting messages for {agent_id}: {e}")
            return []

    async def broadcast_message(self, message: Dict[str, Any],
                                exclude_agents: Optional[List[str]] = None):
        """Broadcast a message to all agents"""
        try:
            # Add metadata
            message['sent_at'] = datetime.utcnow().isoformat()
            message['broker_id'] = f"broadcast_{datetime.utcnow().timestamp()}"

            # Get all agent queues
            pattern = "agent_queue:*"
            agent_queues = await self.redis_client.keys(pattern)

            exclude_agents = exclude_agents or []
            exclude_queues = [f"agent_queue:{agent}" for agent in
                              exclude_agents]

            message_str = json.dumps(message)

            for queue_key in agent_queues:
                if queue_key not in exclude_queues:
                    await self.redis_client.lpush(queue_key, message_str)
                    await self.redis_client.expire(queue_key, 86400)

            logger.debug(
                f"Broadcast message: {message['type']} to {len(agent_queues)} agents")

        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event to all subscribers"""
        try:
            event = {
                'event_type': event_type,
                'data': event_data,
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': f"event_{datetime.utcnow().timestamp()}"
            }

            channel = f"events:{event_type}"
            event_str = json.dumps(event)

            await self.redis_client.publish(channel, event_str)

            logger.debug(f"Published event: {event_type}")

        except Exception as e:
            logger.error(f"Error publishing event: {e}")

    async def subscribe_to_events(self, agent_id: str, event_types: List[str]):
        """Subscribe an agent to specific event types"""
        try:
            if not self.pubsub:
                self.pubsub = self.redis_client.pubsub()

            channels = [f"events:{event_type}" for event_type in event_types]
            await self.pubsub.subscribe(*channels)

            self.subscriptions[agent_id] = event_types

            logger.info(
                f"Agent {agent_id} subscribed to events: {event_types}")

        except Exception as e:
            logger.error(f"Error subscribing to events: {e}")

    async def get_events(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get events for a subscribed agent"""
        try:
            if not self.pubsub or agent_id not in self.subscriptions:
                return []

            events = []

            # Non-blocking get messages
            try:
                while True:
                    message = await asyncio.wait_for(
                        self.pubsub.get_message(
                            ignore_subscribe_messages=True),
                        timeout=0.1
                    )

                    if message and message['type'] == 'message':
                        try:
                            event = json.loads(message['data'])
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
                    else:
                        break

            except asyncio.TimeoutError:
                pass  # No more messages

            return events

        except Exception as e:
            logger.error(f"Error getting events for {agent_id}: {e}")
            return []

    async def store_message_history(self, message: Dict[str, Any]):
        """Store message for debugging/audit purposes"""
        try:
            history_key = "message_history"
            message_str = json.dumps(message)

            # Store in a list with timestamp
            await self.redis_client.lpush(history_key, message_str)

            # Keep only last 1000 messages
            await self.redis_client.ltrim(history_key, 0, 999)

            # Set expiration (7 days)
            await self.redis_client.expire(history_key, 604800)

        except Exception as e:
            logger.error(f"Error storing message history: {e}")

    async def get_message_history(self, limit: int = 100) -> List[
        Dict[str, Any]]:
        """Get message history for debugging"""
        try:
            history_key = "message_history"
            messages_str = await self.redis_client.lrange(history_key, 0,
                                                          limit - 1)

            messages = []
            for msg_str in messages_str:
                try:
                    message = json.loads(msg_str)
                    messages.append(message)
                except json.JSONDecodeError:
                    continue

            return messages

        except Exception as e:
            logger.error(f"Error getting message history: {e}")
            return []

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about message queues"""
        try:
            stats = {}

            # Get all agent queues
            pattern = "agent_queue:*"
            agent_queues = await self.redis_client.keys(pattern)

            for queue_key in agent_queues:
                agent_id = queue_key.split(':')[1]
                queue_length = await self.redis_client.llen(queue_key)

                stats[agent_id] = {
                    'queue_length': queue_length,
                    'queue_key': queue_key
                }

            # Add total stats
            stats['total_queues'] = len(agent_queues)
            stats['total_messages'] = sum(
                s['queue_length'] for s in stats.values() if
                isinstance(s, dict))

            return stats

        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}

    async def clear_agent_queue(self, agent_id: str) -> bool:
        """Clear all messages for a specific agent"""
        try:
            queue_key = f"agent_queue:{agent_id}"
            deleted = await self.redis_client.delete(queue_key)

            logger.info(f"Cleared queue for agent {agent_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error clearing queue for {agent_id}: {e}")
            return False

    async def send_priority_message(self, recipient: str,
                                    message: Dict[str, Any]):
        """Send a high-priority message (goes to front of queue)"""
        try:
            # Add metadata
            message['sent_at'] = datetime.utcnow().isoformat()
            message['broker_id'] = f"priority_{datetime.utcnow().timestamp()}"
            message['priority'] = 'high'

            # Store message at front of recipient's queue
            queue_key = f"agent_queue:{recipient}"
            message_str = json.dumps(message)

            await self.redis_client.rpush(queue_key,
                                          message_str)  # Front of queue

            # Set expiration for queue (24 hours)
            await self.redis_client.expire(queue_key, 86400)

            logger.debug(
                f"Sent priority message to {recipient}: {message['type']}")

        except Exception as e:
            logger.error(f"Error sending priority message to {recipient}: {e}")

    async def health_check(self) -> bool:
        """Check if message broker is healthy"""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Message broker health check failed: {e}")
            return False

    async def close(self):
        """Close message broker connections"""
        try:
            if self.pubsub:
                await self.pubsub.close()

            if self.redis_client:
                await self.redis_client.close()

            logger.info("Closed message broker connections")

        except Exception as e:
            logger.error(f"Error closing message broker: {e}")