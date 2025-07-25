from elasticsearch import Elasticsearch
from config.settings import settings
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ElasticsearchConfig:
    """Elasticsearch configuration and index management"""

    def __init__(self):
        self.client = self._create_client()
        self.indices = {
            "energy_patterns": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_energy_patterns",
            "device_states": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_device_states",
            "user_preferences": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_user_preferences",
            "price_data": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_price_data",
            "weather_data": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_weather_data",
            "agent_decisions": f"{settings.ELASTICSEARCH_INDEX_PREFIX}_agent_decisions"
        }

    def _create_client(self) -> Elasticsearch:
        """Create Elasticsearch client (Elasticsearch 8.x with compatibility mode)"""
        es_config = {
            "hosts": [
                f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
            "request_timeout": 30,
            "retry_on_timeout": True,
            "max_retries": 3,
            "headers": {
                "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
            }
        }

        if settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD:
            es_config["basic_auth"] = (
                settings.ELASTICSEARCH_USERNAME,
                settings.ELASTICSEARCH_PASSWORD
            )

        return Elasticsearch(**es_config)

    def create_indices(self):
        """Create all required indices with proper mappings"""
        indices_mappings = {
            "energy_patterns": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "device_id": {"type": "keyword"},
                        "energy_consumption": {"type": "float"},
                        "pattern_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        },
                        "pattern_description": {"type": "text"},
                        "efficiency_score": {"type": "float"}
                    }
                }
            },
            "device_states": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "device_id": {"type": "keyword"},
                        "device_type": {"type": "keyword"},
                        "state": {"type": "object"},
                        "state_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        },
                        "energy_consumption": {"type": "float"},
                        "room": {"type": "keyword"}
                    }
                }
            },
            "user_preferences": {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "preference_type": {"type": "keyword"},
                        "preferences": {"type": "object"},
                        "preference_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        },
                        "priority_score": {"type": "float"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            },
            "price_data": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "price_per_kwh": {"type": "float"},
                        "price_tier": {"type": "keyword"},
                        "utility_company": {"type": "keyword"},
                        "forecast": {"type": "boolean"},
                        "price_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        }
                    }
                }
            },
            "weather_data": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "temperature": {"type": "float"},
                        "humidity": {"type": "float"},
                        "solar_radiation": {"type": "float"},
                        "weather_conditions": {"type": "keyword"},
                        "weather_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        }
                    }
                }
            },
            "agent_decisions": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "agent_id": {"type": "keyword"},
                        "decision_type": {"type": "keyword"},
                        "decision_data": {"type": "object"},
                        "reasoning": {"type": "text"},
                        "decision_embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIMENSION
                        },
                        "confidence_score": {"type": "float"},
                        "execution_status": {"type": "keyword"}
                    }
                }
            }
        }

        for index_name, mapping in indices_mappings.items():
            full_index_name = self.indices[index_name]
            if not self.client.indices.exists(index=full_index_name):
                self.client.indices.create(index=full_index_name, body=mapping)
                logger.info(f"Created index: {full_index_name}")
            else:
                logger.info(f"Index already exists: {full_index_name}")

    def get_client(self) -> Elasticsearch:
        """Get Elasticsearch client instance"""
        return self.client

    def health_check(self) -> bool:
        """Check if Elasticsearch is healthy"""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False


# Global Elasticsearch configuration instance
es_config = ElasticsearchConfig()