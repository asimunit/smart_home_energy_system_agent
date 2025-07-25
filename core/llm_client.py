import asyncio
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Gemini LLM"""

    def __init__(self):
        self.model_name = settings.GEMINI_MODEL
        self._configure_gemini()
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(
            f"Initialized Gemini LLM client with model: {self.model_name}")

    def _configure_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=settings.GEMINI_API_KEY)

    async def generate_response(self,
                                prompt: str,
                                temperature: float = 0.7,
                                max_tokens: int = 1000,
                                system_instruction: Optional[
                                    str] = None) -> str:
        """Generate response from Gemini"""
        try:
            # Add system instruction to prompt if provided
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            else:
                full_prompt = prompt

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1
            )

            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config
            )

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("No response generated from Gemini")
                return "No response generated"

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise

    async def analyze_energy_pattern(self,
                                     energy_data: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[
        str, Any]:
        """Analyze energy consumption patterns"""
        prompt = f"""
        Analyze the following energy consumption pattern for a smart home device:

        Energy Data:
        {energy_data}

        Context Information:
        {context}

        Please provide analysis in the following JSON format:
        {{
            "efficiency_score": <0-100>,
            "pattern_type": "<normal|peak|wasteful|optimal>",
            "recommendations": ["recommendation1", "recommendation2"],
            "anomalies": ["anomaly1", "anomaly2"],
            "predicted_savings": <percentage>,
            "confidence": <0-1>
        }}

        Consider factors like:
        - Time of use patterns
        - Energy pricing tiers
        - Weather conditions
        - User behavior patterns
        - Device specifications
        """

        system_instruction = "You are an expert energy efficiency analyst for smart home systems. Provide accurate, actionable insights based on data patterns."

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800,
            system_instruction=system_instruction
        )

        return self._parse_json_response(response)

    async def optimize_device_schedule(self,
                                       device_info: Dict[str, Any],
                                       preferences: Dict[str, Any],
                                       constraints: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate optimal device scheduling"""
        prompt = f"""
        Create an optimal schedule for the following smart home device:

        Device Information:
        {device_info}

        User Preferences:
        {preferences}

        System Constraints:
        {constraints}

        Generate a schedule in the following JSON format:
        {{
            "schedule": [
                {{
                    "time": "HH:MM",
                    "action": "<on|off|dim|temperature_change>",
                    "value": <numeric_value_if_applicable>,
                    "reason": "<brief_explanation>"
                }}
            ],
            "expected_savings": <percentage>,
            "comfort_impact": "<minimal|moderate|significant>",
            "confidence": <0-1>
        }}

        Optimize for:
        - Energy cost reduction
        - User comfort maintenance
        - Peak demand avoidance
        - Device longevity
        """

        system_instruction = "You are a smart home automation expert specializing in energy-efficient device scheduling."

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1000,
            system_instruction=system_instruction
        )

        return self._parse_json_response(response)

    async def negotiate_priorities(self,
                                   conflicting_requests: List[Dict[str, Any]],
                                   system_state: Dict[str, Any]) -> Dict[
        str, Any]:
        """Negotiate between conflicting agent requests"""
        prompt = f"""
        You are a negotiator agent in a smart home energy management system.
        Multiple agents have conflicting requests that need to be resolved.

        Conflicting Requests:
        {conflicting_requests}

        Current System State:
        {system_state}

        Please provide a negotiation solution in JSON format:
        {{
            "resolution": {{
                "agent_id": "<winner_agent_id>",
                "compromise_level": <0-100>,
                "modifications": [
                    {{
                        "agent_id": "<agent_id>",
                        "original_request": "<description>",
                        "modified_request": "<description>",
                        "reason": "<explanation>"
                    }}
                ]
            }},
            "justification": "<detailed_reasoning>",
            "alternative_solutions": ["solution1", "solution2"],
            "confidence": <0-1>
        }}

        Consider:
        - Energy efficiency priorities
        - User comfort requirements
        - System safety constraints
        - Cost implications
        - Fairness between agents
        """

        system_instruction = "You are an expert mediator for smart home systems, skilled in balancing competing requirements while optimizing overall system performance."

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1200,
            system_instruction=system_instruction
        )

        return self._parse_json_response(response)

    async def predict_energy_demand(self,
                                    historical_data: Dict[str, Any],
                                    external_factors: Dict[str, Any]) -> Dict[
        str, Any]:
        """Predict future energy demand"""
        prompt = f"""
        Predict energy demand for the next 24 hours based on historical patterns and external factors.

        Historical Data:
        {historical_data}

        External Factors:
        {external_factors}

        Provide prediction in JSON format:
        {{
            "hourly_predictions": [
                {{
                    "hour": <0-23>,
                    "predicted_consumption": <kwh>,
                    "confidence": <0-1>,
                    "peak_probability": <0-1>
                }}
            ],
            "daily_total": <kwh>,
            "peak_hours": [<hour_list>],
            "optimization_opportunities": [
                {{
                    "time_window": "<start_hour>-<end_hour>",
                    "potential_savings": <percentage>,
                    "strategy": "<description>"
                }}
            ],
            "confidence": <0-1>
        }}

        Consider:
        - Seasonal patterns
        - Weather forecasts
        - Historical usage patterns
        - Special events or holidays
        - Device maintenance schedules
        """

        system_instruction = "You are an energy demand forecasting expert with deep knowledge of residential energy consumption patterns."

        response = await self.generate_response(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500,
            system_instruction=system_instruction
        )

        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            import json

            # Find JSON content in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return {"error": "No JSON found in response",
                        "raw_response": response}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {e}")
            return {"error": "JSON parsing failed", "raw_response": response}
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return {"error": "Unexpected parsing error",
                    "raw_response": response}

    async def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            test_response = await self.generate_response(
                prompt="Respond with 'OK' if you can receive this message.",
                temperature=0.1,
                max_tokens=10
            )

            return "OK" in test_response.upper()

        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False