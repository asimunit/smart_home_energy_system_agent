import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4

from core.llm_client import LLMClient
from core.embedding_service import EmbeddingService
from core.database import DatabaseService
from core.message_broker import MessageBroker
from config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the smart home energy system"""

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.is_running = False
        self.last_update = None

        # Core services
        self.llm_client = LLMClient()
        self.embedding_service = EmbeddingService()
        self.db_service = DatabaseService()
        self.message_broker = MessageBroker()

        # Agent state
        self.state = {}
        self.memory = []
        self.decision_history = []

        logger.info(f"Initialized {self.agent_type} agent: {self.agent_id}")

    async def start(self):
        """Start the agent"""
        self.is_running = True
        await self.initialize()
        logger.info(f"Started agent: {self.agent_id}")

        # Start main processing loop
        asyncio.create_task(self.run())

    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        await self.cleanup()
        logger.info(f"Stopped agent: {self.agent_id}")

    async def run(self):
        """Main agent processing loop"""
        while self.is_running:
            try:
                # Process incoming messages
                await self.process_messages()

                # Execute agent-specific logic
                await self.execute()

                # Update agent state
                await self.update_state()

                # Sleep for update interval
                await asyncio.sleep(settings.AGENT_UPDATE_INTERVAL)

            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")
                await asyncio.sleep(settings.AGENT_UPDATE_INTERVAL)

    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources"""
        pass

    @abstractmethod
    async def execute(self):
        """Execute agent-specific logic"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup agent-specific resources"""
        pass

    async def process_messages(self):
        """Process incoming messages from other agents"""
        messages = await self.message_broker.get_messages(self.agent_id)

        for message in messages:
            await self.handle_message(message)

    async def handle_message(self, message: Dict[str, Any]):
        """Handle a single message"""
        try:
            message_type = message.get("type")
            sender = message.get("sender")
            data = message.get("data", {})

            logger.debug(
                f"Agent {self.agent_id} received {message_type} from {sender}")

            # Process message based on type
            if message_type == "request":
                await self.handle_request(sender, data)
            elif message_type == "response":
                await self.handle_response(sender, data)
            elif message_type == "notification":
                await self.handle_notification(sender, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle request messages - to be overridden by subclasses"""
        pass

    async def handle_response(self, sender: str, data: Dict[str, Any]):
        """Handle response messages - to be overridden by subclasses"""
        pass

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notification messages - to be overridden by subclasses"""
        pass

    async def send_message(self, recipient: str, message_type: str,
                           data: Dict[str, Any]):
        """Send a message to another agent"""
        message = {
            "id": str(uuid4()),
            "sender": self.agent_id,
            "recipient": recipient,
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.message_broker.send_message(recipient, message)

    async def broadcast_message(self, message_type: str, data: Dict[str, Any],
                                exclude_agents: Optional[List[str]] = None):
        """Broadcast a message to all agents"""
        message = {
            "id": str(uuid4()),
            "sender": self.agent_id,
            "recipient": "broadcast",
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.message_broker.broadcast_message(message, exclude_agents)

    async def make_decision(self, context: Dict[str, Any],
                            options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision using LLM"""
        prompt = self.build_decision_prompt(context, options)

        try:
            response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )

            decision = self.parse_decision_response(response)

            # Store decision in history
            decision_record = {
                "timestamp": datetime.utcnow(),
                "context": context,
                "options": options,
                "decision": decision,
                "reasoning": response
            }

            self.decision_history.append(decision_record)

            # Store in database with embedding
            await self.store_decision(decision_record)

            return decision

        except Exception as e:
            logger.error(f"Error making decision in {self.agent_id}: {e}")
            return self.get_default_decision(options)

    def build_decision_prompt(self, context: Dict[str, Any],
                              options: List[Dict[str, Any]]) -> str:
        """Build prompt for decision making - to be overridden by subclasses"""
        prompt = f"""
        You are a {self.agent_type} agent in a smart home energy management system.

        Current Context:
        {json.dumps(context, indent=2)}

        Available Options:
        {json.dumps(options, indent=2)}

        Please analyze the situation and choose the best option considering:
        1. Energy efficiency
        2. Cost optimization
        3. User comfort
        4. System reliability

        Respond with your decision in JSON format:
        {{
            "chosen_option": <index_of_chosen_option>,
            "confidence": <confidence_score_0_to_1>,
            "reasoning": "<brief_explanation>"
        }}
        """

        return prompt

    def parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into decision format"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error(f"Error parsing decision response: {e}")
            return {
                "chosen_option": 0,
                "confidence": 0.5,
                "reasoning": "Default choice due to parsing error"
            }

    def get_default_decision(self, options: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """Get default decision when LLM fails"""
        return {
            "chosen_option": 0,
            "confidence": 0.3,
            "reasoning": "Default choice due to LLM failure"
        }

    async def store_decision(self, decision_record: Dict[str, Any]):
        """Store decision in database with embedding"""
        try:
            # Generate embedding for decision context
            decision_text = f"{decision_record['reasoning']} Context: {json.dumps(decision_record['context'])}"
            embedding = await self.embedding_service.generate_embedding(
                decision_text)

            # Store in Elasticsearch
            doc = {
                "timestamp": decision_record["timestamp"],
                "agent_id": self.agent_id,
                "decision_type": self.agent_type,
                "decision_data": decision_record["decision"],
                "reasoning": decision_record["reasoning"],
                "decision_embedding": embedding,
                "confidence_score": decision_record["decision"].get(
                    "confidence", 0.5),
                "execution_status": "planned"
            }

            await self.db_service.store_document("agent_decisions", doc)

        except Exception as e:
            logger.error(f"Error storing decision: {e}")

    async def update_state(self):
        """Update agent state"""
        self.last_update = datetime.utcnow()
        self.state["last_update"] = self.last_update.isoformat()
        self.state["is_running"] = self.is_running

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "state": self.state,
            "decision_count": len(self.decision_history)
        }