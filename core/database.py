import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from config.elasticsearch_config import es_config
from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for interacting with Elasticsearch database"""

    def __init__(self):
        self.es_client = self._create_async_client()
        self.indices = es_config.indices
        logger.info("Initialized database service")

    def _create_async_client(self) -> AsyncElasticsearch:
        """Create async Elasticsearch client"""
        es_config_dict = {
            "hosts": [
                f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
            "max_retries": 3,
            "retry_on_timeout": True
        }

        if settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD:
            es_config_dict["basic_auth"] = (
                settings.ELASTICSEARCH_USERNAME,
                settings.ELASTICSEARCH_PASSWORD
            )

        return AsyncElasticsearch(**es_config_dict)

    async def store_document(self, index_type: str, document: Dict[str, Any],
                             doc_id: Optional[str] = None) -> bool:
        """Store a document in Elasticsearch"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return False

            # Add timestamp if not present
            if 'timestamp' not in document:
                document['timestamp'] = datetime.utcnow()

            # Store document
            if doc_id:
                response = await self.es_client.index(
                    index=index_name,
                    id=doc_id,
                    body=document
                )
            else:
                response = await self.es_client.index(
                    index=index_name,
                    body=document
                )

            logger.debug(f"Stored document in {index_name}: {response['_id']}")
            return response['result'] in ['created', 'updated']

        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False

    async def get_document(self, index_type: str, doc_id: str) -> Optional[
        Dict[str, Any]]:
        """Get a document by ID"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return None

            response = await self.es_client.get(
                index=index_name,
                id=doc_id
            )

            return response['_source']

        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None

    async def search_documents(self, index_type: str, query: Dict[str, Any],
                               size: int = 100) -> List[Dict[str, Any]]:
        """Search documents in an index"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return []

            response = await self.es_client.search(
                index=index_name,
                body={"query": query, "size": size}
            )

            return [hit['_source'] for hit in response['hits']['hits']]

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def vector_search(self, index_type: str, query_vector: List[float],
                            field_name: str, size: int = 10,
                            min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return []

            query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc[params.field]) + 1.0",
                        "params": {
                            "query_vector": query_vector,
                            "field": field_name
                        }
                    }
                }
            }

            response = await self.es_client.search(
                index=index_name,
                body={
                    "query": query,
                    "size": size,
                    "min_score": min_score
                }
            )

            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['_score'] = hit['_score']
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

    async def store_energy_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Store energy pattern with embedding"""
        return await self.store_document("energy_patterns", pattern_data)

    async def store_device_state(self, device_data: Dict[str, Any]) -> bool:
        """Store device state with embedding"""
        return await self.store_document("device_states", device_data)

    async def store_user_preferences(self,
                                     preferences_data: Dict[str, Any]) -> bool:
        """Store user preferences with embedding"""
        return await self.store_document("user_preferences", preferences_data)

    async def store_price_data(self, price_data: Dict[str, Any]) -> bool:
        """Store energy price data with embedding"""
        return await self.store_document("price_data", price_data)

    async def store_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Store weather data with embedding"""
        return await self.store_document("weather_data", weather_data)

    async def find_similar_energy_patterns(self, query_embedding: List[float],
                                           size: int = 10) -> List[
        Dict[str, Any]]:
        """Find similar energy consumption patterns"""
        return await self.vector_search(
            "energy_patterns",
            query_embedding,
            "pattern_embedding",
            size
        )

    async def find_similar_device_states(self, query_embedding: List[float],
                                         device_type: Optional[str] = None,
                                         size: int = 10) -> List[
        Dict[str, Any]]:
        """Find similar device states"""
        # Add device type filter if specified
        if device_type:
            query = {
                "bool": {
                    "must": [
                        {"match": {"device_type": device_type}},
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc['state_embedding']) + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ]
                }
            }

            try:
                index_name = self.indices["device_states"]
                response = await self.es_client.search(
                    index=index_name,
                    body={"query": query, "size": size}
                )

                results = []
                for hit in response['hits']['hits']:
                    result = hit['_source']
                    result['_score'] = hit['_score']
                    results.append(result)

                return results

            except Exception as e:
                logger.error(f"Error finding similar device states: {e}")
                return []
        else:
            return await self.vector_search(
                "device_states",
                query_embedding,
                "state_embedding",
                size
            )

    async def get_recent_energy_data(self, device_id: Optional[str] = None,
                                     hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent energy consumption data"""
        try:
            query = {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": f"now-{hours}h"
                                }
                            }
                        }
                    ]
                }
            }

            if device_id:
                query["bool"]["must"].append(
                    {"term": {"device_id": device_id}})

            return await self.search_documents("energy_patterns", query,
                                               size=1000)

        except Exception as e:
            logger.error(f"Error getting recent energy data: {e}")
            return []

    async def get_user_preferences(self, user_id: str) -> Optional[
        Dict[str, Any]]:
        """Get user preferences by user ID"""
        try:
            query = {"term": {"user_id": user_id}}
            results = await self.search_documents("user_preferences", query,
                                                  size=1)

            return results[0] if results else None

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None

    async def get_current_price_data(self) -> Optional[Dict[str, Any]]:
        """Get current energy price data"""
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"forecast": False}},
                        {
                            "range": {
                                "timestamp": {
                                    "gte": "now-1h"
                                }
                            }
                        }
                    ]
                }
            }

            # Sort by timestamp descending to get most recent
            body = {
                "query": query,
                "size": 1,
                "sort": [{"timestamp": {"order": "desc"}}]
            }

            index_name = self.indices["price_data"]
            response = await self.es_client.search(index=index_name, body=body)

            hits = response['hits']['hits']
            return hits[0]['_source'] if hits else None

        except Exception as e:
            logger.error(f"Error getting current price data: {e}")
            return None

    async def get_agent_decisions(self, agent_id: str, hours: int = 24) -> \
    List[Dict[str, Any]]:
        """Get recent decisions by an agent"""
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"agent_id": agent_id}},
                        {
                            "range": {
                                "timestamp": {
                                    "gte": f"now-{hours}h"
                                }
                            }
                        }
                    ]
                }
            }

            return await self.search_documents("agent_decisions", query,
                                               size=100)

        except Exception as e:
            logger.error(f"Error getting agent decisions: {e}")
            return []

    async def update_document(self, index_type: str, doc_id: str,
                              updates: Dict[str, Any]) -> bool:
        """Update a document"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return False

            response = await self.es_client.update(
                index=index_name,
                id=doc_id,
                body={"doc": updates}
            )

            return response['result'] == 'updated'

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    async def delete_document(self, index_type: str, doc_id: str) -> bool:
        """Delete a document"""
        try:
            index_name = self.indices.get(index_type)
            if not index_name:
                logger.error(f"Unknown index type: {index_type}")
                return False

            response = await self.es_client.delete(
                index=index_name,
                id=doc_id
            )

            return response['result'] == 'deleted'

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            health = await self.es_client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self):
        """Close database connections"""
        try:
            await self.es_client.close()
            logger.info("Closed database connections")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")