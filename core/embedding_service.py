import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using mxbai model"""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = None
        self._initialize_model()
        logger.info(
            f"Initialized embedding service with model: {self.model_name}")

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(
                f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension

        try:
            # Run embedding generation in thread pool to avoid blocking
            embedding = await asyncio.to_thread(
                self.model.encode,
                text,
                normalize_embeddings=True
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension

    async def generate_embeddings_batch(self, texts: List[str]) -> List[
        List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return [[0.0] * self.dimension] * len(texts)

        try:
            # Run batch embedding generation in thread pool
            embeddings = await asyncio.to_thread(
                self.model.encode,
                valid_texts,
                normalize_embeddings=True,
                batch_size=32
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [[0.0] * self.dimension] * len(texts)

    async def embed_energy_pattern(self, pattern_data: Dict[str, Any]) -> List[
        float]:
        """Generate embedding for energy consumption pattern"""
        try:
            # Create descriptive text from pattern data
            text_parts = []

            if 'device_type' in pattern_data:
                text_parts.append(f"Device: {pattern_data['device_type']}")

            if 'consumption' in pattern_data:
                text_parts.append(
                    f"Consumption: {pattern_data['consumption']} kWh")

            if 'time_period' in pattern_data:
                text_parts.append(f"Time: {pattern_data['time_period']}")

            if 'efficiency_score' in pattern_data:
                text_parts.append(
                    f"Efficiency: {pattern_data['efficiency_score']}")

            if 'room' in pattern_data:
                text_parts.append(f"Location: {pattern_data['room']}")

            if 'weather_conditions' in pattern_data:
                text_parts.append(
                    f"Weather: {pattern_data['weather_conditions']}")

            pattern_text = " | ".join(text_parts)
            return await self.generate_embedding(pattern_text)

        except Exception as e:
            logger.error(f"Error embedding energy pattern: {e}")
            return [0.0] * self.dimension

    async def embed_device_state(self, device_data: Dict[str, Any]) -> List[
        float]:
        """Generate embedding for device state"""
        try:
            text_parts = []

            if 'device_id' in device_data:
                text_parts.append(f"Device ID: {device_data['device_id']}")

            if 'device_type' in device_data:
                text_parts.append(f"Type: {device_data['device_type']}")

            if 'state' in device_data:
                state_str = str(device_data['state'])
                text_parts.append(f"State: {state_str}")

            if 'power_consumption' in device_data:
                text_parts.append(
                    f"Power: {device_data['power_consumption']}W")

            if 'room' in device_data:
                text_parts.append(f"Room: {device_data['room']}")

            device_text = " | ".join(text_parts)
            return await self.generate_embedding(device_text)

        except Exception as e:
            logger.error(f"Error embedding device state: {e}")
            return [0.0] * self.dimension

    async def embed_user_preferences(self, preferences: Dict[str, Any]) -> \
    List[float]:
        """Generate embedding for user preferences"""
        try:
            text_parts = []

            if 'comfort_level' in preferences:
                text_parts.append(
                    f"Comfort Level: {preferences['comfort_level']}")

            if 'temperature_preference' in preferences:
                temp_pref = preferences['temperature_preference']
                text_parts.append(
                    f"Temperature: {temp_pref.get('min', 'auto')}-{temp_pref.get('max', 'auto')}°F")

            if 'energy_priority' in preferences:
                text_parts.append(
                    f"Energy Priority: {preferences['energy_priority']}")

            if 'schedule' in preferences:
                text_parts.append(f"Schedule: {preferences['schedule']}")

            if 'cost_sensitivity' in preferences:
                text_parts.append(
                    f"Cost Sensitivity: {preferences['cost_sensitivity']}")

            preferences_text = " | ".join(text_parts)
            return await self.generate_embedding(preferences_text)

        except Exception as e:
            logger.error(f"Error embedding user preferences: {e}")
            return [0.0] * self.dimension

    async def embed_price_data(self, price_data: Dict[str, Any]) -> List[
        float]:
        """Generate embedding for energy price data"""
        try:
            text_parts = []

            if 'price_per_kwh' in price_data:
                text_parts.append(f"Price: ${price_data['price_per_kwh']}/kWh")

            if 'price_tier' in price_data:
                text_parts.append(f"Tier: {price_data['price_tier']}")

            if 'time_of_day' in price_data:
                text_parts.append(f"Time: {price_data['time_of_day']}")

            if 'demand_level' in price_data:
                text_parts.append(f"Demand: {price_data['demand_level']}")

            if 'utility_company' in price_data:
                text_parts.append(f"Utility: {price_data['utility_company']}")

            price_text = " | ".join(text_parts)
            return await self.generate_embedding(price_text)

        except Exception as e:
            logger.error(f"Error embedding price data: {e}")
            return [0.0] * self.dimension

    async def embed_weather_data(self, weather_data: Dict[str, Any]) -> List[
        float]:
        """Generate embedding for weather data"""
        try:
            text_parts = []

            if 'temperature' in weather_data:
                text_parts.append(
                    f"Temperature: {weather_data['temperature']}°F")

            if 'humidity' in weather_data:
                text_parts.append(f"Humidity: {weather_data['humidity']}%")

            if 'conditions' in weather_data:
                text_parts.append(f"Conditions: {weather_data['conditions']}")

            if 'solar_radiation' in weather_data:
                text_parts.append(
                    f"Solar: {weather_data['solar_radiation']} W/m²")

            if 'wind_speed' in weather_data:
                text_parts.append(f"Wind: {weather_data['wind_speed']} mph")

            weather_text = " | ".join(text_parts)
            return await self.generate_embedding(weather_text)

        except Exception as e:
            logger.error(f"Error embedding weather data: {e}")
            return [0.0] * self.dimension

    async def compute_similarity(self, embedding1: List[float],
                                 embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    async def find_similar_patterns(self, query_embedding: List[float],
                                    pattern_embeddings: List[List[float]],
                                    threshold: float = 0.7) -> List[int]:
        """Find similar patterns based on embedding similarity"""
        try:
            similar_indices = []

            for i, pattern_embedding in enumerate(pattern_embeddings):
                similarity = await self.compute_similarity(query_embedding,
                                                           pattern_embedding)
                if similarity >= threshold:
                    similar_indices.append(i)

            return similar_indices

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": getattr(self.model, 'max_seq_length',
                                           'unknown'),
            "device": str(self.model.device) if self.model else 'unknown'
        }

    async def health_check(self) -> bool:
        """Check if embedding service is healthy"""
        try:
            test_embedding = await self.generate_embedding("test text")
            return len(test_embedding) == self.dimension
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False