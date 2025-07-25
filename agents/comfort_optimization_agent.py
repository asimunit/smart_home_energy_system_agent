import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from core.base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class ComfortOptimizationAgent(BaseAgent):
    """Agent responsible for optimizing user comfort while maintaining energy efficiency"""

    def __init__(self):
        super().__init__("comfort_optimization", "ComfortOptimization")
        self.user_preferences = {}
        self.comfort_metrics = {}
        self.occupancy_patterns = {}
        self.comfort_zones = {}
        self.learning_enabled = True
        self.comfort_priority = 'balanced'  # balanced, comfort_first, efficiency_first

    async def initialize(self):
        """Initialize the comfort optimization agent"""
        logger.info("Initializing Comfort Optimization Agent")

        # Load user preferences and comfort profiles
        await self._load_user_preferences()

        # Initialize comfort zones and metrics
        await self._initialize_comfort_zones()

        # Subscribe to relevant events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            [
                'occupancy_change', 'temperature_change', 'humidity_change',
                'lighting_change', 'user_feedback', 'schedule_change',
                'weather_update', 'energy_emergency'
            ]
        )

        # Initialize learning system
        await self._initialize_comfort_learning()

        logger.info("Comfort Optimization Agent initialized successfully")

    async def execute(self):
        """Main execution logic for comfort optimization"""
        try:
            # Monitor current comfort conditions
            await self._monitor_comfort_conditions()

            # Analyze occupancy patterns
            await self._analyze_occupancy_patterns()

            # Optimize comfort settings
            await self._optimize_comfort_settings()

            # Learn from user behavior and feedback
            if self.learning_enabled:
                await self._learn_from_user_behavior()

            # Generate comfort recommendations
            await self._generate_comfort_recommendations()

            # Handle comfort conflicts and priorities
            await self._resolve_comfort_conflicts()

        except Exception as e:
            logger.error(f"Error in comfort optimization execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Comfort Optimization Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents or users"""
        request_type = data.get('request_type')

        if request_type == 'get_comfort_preferences':
            await self._handle_comfort_preferences_request(sender, data)
        elif request_type == 'set_comfort_preferences':
            await self._handle_set_preferences_request(sender, data)
        elif request_type == 'get_comfort_status':
            await self._handle_comfort_status_request(sender, data)
        elif request_type == 'optimize_for_comfort':
            await self._handle_optimize_comfort_request(sender, data)
        elif request_type == 'learn_user_preference':
            await self._handle_learn_preference_request(sender, data)
        elif request_type == 'set_comfort_priority':
            await self._handle_set_priority_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'occupancy_change':
            await self._handle_occupancy_change(sender, data)
        elif notification_type == 'temperature_change':
            await self._handle_temperature_change(sender, data)
        elif notification_type == 'humidity_change':
            await self._handle_humidity_change(sender, data)
        elif notification_type == 'lighting_change':
            await self._handle_lighting_change(sender, data)
        elif notification_type == 'user_feedback':
            await self._handle_user_feedback(sender, data)
        elif notification_type == 'weather_update':
            await self._handle_weather_update(sender, data)
        elif notification_type == 'energy_emergency':
            await self._handle_energy_emergency(sender, data)

    async def _load_user_preferences(self):
        """Load user comfort preferences from database"""
        try:
            # Get user preferences from database
            preferences = await self.db_service.get_user_preferences(
                'default_user')

            if preferences:
                self.user_preferences = preferences.get('preferences', {})
            else:
                # Initialize default preferences
                self.user_preferences = {
                    'temperature': {
                        'heating_setpoint': 70,  # °F
                        'cooling_setpoint': 76,  # °F
                        'tolerance': 2,  # ±2°F
                        'schedule': {
                            'weekday': {
                                'wake': {'time': '06:30', 'temp': 72},
                                'leave': {'time': '08:00', 'temp': 78},
                                'return': {'time': '18:00', 'temp': 74},
                                'sleep': {'time': '22:00', 'temp': 68}
                            },
                            'weekend': {
                                'wake': {'time': '08:00', 'temp': 72},
                                'sleep': {'time': '23:00', 'temp': 68}
                            }
                        }
                    },
                    'lighting': {
                        'brightness_preference': 75,  # 0-100%
                        'color_temperature_day': 4000,  # K
                        'color_temperature_evening': 2700,  # K
                        'auto_adjust': True,
                        'circadian_lighting': True
                    },
                    'humidity': {
                        'target_range': {'min': 40, 'max': 60},  # %
                        'tolerance': 10  # ±10%
                    },
                    'air_quality': {
                        'ventilation_preference': 'auto',
                        'air_circulation': True
                    },
                    'energy_comfort_balance': 'balanced',
                    # balanced, comfort_first, efficiency_first
                    'learning_enabled': True,
                    'occupancy_based_optimization': True
                }

            logger.info("User comfort preferences loaded")

        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")

    async def _initialize_comfort_zones(self):
        """Initialize comfort zones for different areas"""
        try:
            # Define comfort zones (rooms/areas with specific comfort profiles)
            self.comfort_zones = {
                'living_room': {
                    'temperature_range': {'min': 70, 'max': 76},
                    'humidity_range': {'min': 40, 'max': 60},
                    'lighting_preference': 'adaptive',
                    'priority': 'high',
                    'occupancy_sensors': ['living_room_sensor'],
                    'current_conditions': {}
                },
                'master_bedroom': {
                    'temperature_range': {'min': 68, 'max': 74},
                    'humidity_range': {'min': 35, 'max': 55},
                    'lighting_preference': 'warm',
                    'priority': 'high',
                    'occupancy_sensors': ['bedroom_sensor'],
                    'current_conditions': {}
                },
                'kitchen': {
                    'temperature_range': {'min': 68, 'max': 78},
                    'humidity_range': {'min': 30, 'max': 70},
                    'lighting_preference': 'bright',
                    'priority': 'medium',
                    'occupancy_sensors': ['kitchen_sensor'],
                    'current_conditions': {}
                },
                'office': {
                    'temperature_range': {'min': 70, 'max': 75},
                    'humidity_range': {'min': 40, 'max': 55},
                    'lighting_preference': 'bright_cool',
                    'priority': 'high',
                    'occupancy_sensors': ['office_sensor'],
                    'current_conditions': {}
                }
            }

            logger.info(f"Initialized {len(self.comfort_zones)} comfort zones")

        except Exception as e:
            logger.error(f"Error initializing comfort zones: {e}")

    async def _initialize_comfort_learning(self):
        """Initialize the comfort learning system"""
        try:
            # Initialize learning parameters
            self.learning_params = {
                'learning_rate': 0.1,
                'feedback_weight': 0.8,
                'pattern_recognition_enabled': True,
                'adaptive_scheduling': True,
                'user_behavior_tracking': True,
                'minimum_data_points': 10
            }

            # Initialize comfort metrics tracking
            self.comfort_metrics = {
                'temperature_satisfaction': [],
                'lighting_satisfaction': [],
                'humidity_satisfaction': [],
                'overall_comfort_score': [],
                'energy_efficiency_score': [],
                'user_interventions': []
            }

            logger.info("Comfort learning system initialized")

        except Exception as e:
            logger.error(f"Error initializing comfort learning: {e}")

    async def _monitor_comfort_conditions(self):
        """Monitor current comfort conditions across all zones"""
        try:
            for zone_id, zone_config in self.comfort_zones.items():
                # Get current environmental conditions for this zone
                conditions = await self._get_zone_conditions(zone_id)

                if conditions:
                    zone_config['current_conditions'] = conditions

                    # Calculate comfort score for this zone
                    comfort_score = await self._calculate_comfort_score(
                        zone_id, conditions)

                    # Store comfort data with embedding
                    comfort_data = {
                        'timestamp': datetime.utcnow(),
                        'zone_id': zone_id,
                        'temperature': conditions.get('temperature', 72),
                        'humidity': conditions.get('humidity', 50),
                        'lighting_brightness': conditions.get(
                            'lighting_brightness', 75),
                        'occupancy': conditions.get('occupancy', False),
                        'comfort_score': comfort_score,
                        'comfort_factors': self._analyze_comfort_factors(
                            zone_id, conditions)
                    }

                    # Generate embedding for comfort pattern
                    comfort_text = f"Zone: {zone_id}, Temp: {conditions.get('temperature', 72)}°F, Humidity: {conditions.get('humidity', 50)}%, Comfort: {comfort_score}"
                    embedding = await self.embedding_service.generate_embedding(
                        comfort_text)
                    comfort_data['pattern_embedding'] = embedding

                    await self.db_service.store_document('user_preferences', {
                        'user_id': 'default_user',
                        'preference_type': 'comfort_monitoring',
                        'preferences': comfort_data,
                        'preference_embedding': embedding,
                        'timestamp': datetime.utcnow()
                    })

        except Exception as e:
            logger.error(f"Error monitoring comfort conditions: {e}")

    async def _get_zone_conditions(self, zone_id: str) -> Dict[str, Any]:
        """Get current environmental conditions for a zone"""
        try:
            conditions = {}

            # Get temperature data from HVAC agent
            try:
                hvac_response = await self._request_from_agent(
                    'hvac', 'get_hvac_status', {'zone': zone_id}
                )
                conditions['temperature'] = hvac_response.get(
                    'current_temperature', 72)
                conditions['humidity'] = hvac_response.get('humidity', 50)
            except:
                conditions['temperature'] = 72
                conditions['humidity'] = 50

            # Get lighting data from lighting agent
            try:
                lighting_response = await self._request_from_agent(
                    'lighting', 'get_lighting_status', {'room': zone_id}
                )
                conditions['lighting_brightness'] = lighting_response.get(
                    'brightness', 75)
                conditions['lighting_color_temp'] = lighting_response.get(
                    'color_temperature', 2700)
            except:
                conditions['lighting_brightness'] = 75
                conditions['lighting_color_temp'] = 2700

            # Simulate occupancy detection (in real system, would come from sensors)
            conditions['occupancy'] = await self._detect_occupancy(zone_id)

            # Add air quality if available
            conditions['air_quality'] = 'good'  # Simplified

            return conditions

        except Exception as e:
            logger.error(f"Error getting zone conditions for {zone_id}: {e}")
            return {}

    async def _detect_occupancy(self, zone_id: str) -> bool:
        """Detect occupancy in a zone"""
        # Simplified occupancy detection
        # In a real system, this would integrate with occupancy sensors
        current_hour = datetime.utcnow().hour

        # Basic schedule-based occupancy simulation
        if zone_id == 'living_room':
            return 18 <= current_hour <= 23  # Evening hours
        elif zone_id == 'master_bedroom':
            return 22 <= current_hour or current_hour <= 7  # Night/early morning
        elif zone_id == 'kitchen':
            return current_hour in [7, 8, 12, 13, 18, 19]  # Meal times
        elif zone_id == 'office':
            return 9 <= current_hour <= 17  # Work hours

        return False

    async def _calculate_comfort_score(self, zone_id: str,
                                       conditions: Dict[str, Any]) -> float:
        """Calculate comfort score for a zone based on current conditions"""
        try:
            zone_config = self.comfort_zones[zone_id]
            user_prefs = self.user_preferences

            score_components = []

            # Temperature comfort score
            temp = conditions.get('temperature', 72)
            temp_range = zone_config['temperature_range']
            temp_tolerance = user_prefs.get('temperature', {}).get('tolerance',
                                                                   2)

            if temp_range['min'] - temp_tolerance <= temp <= temp_range[
                'max'] + temp_tolerance:
                if temp_range['min'] <= temp <= temp_range['max']:
                    temp_score = 1.0  # Perfect
                else:
                    # Gradual decrease in comfort outside ideal range
                    deviation = min(abs(temp - temp_range['min']),
                                    abs(temp - temp_range['max']))
                    temp_score = max(0, 1.0 - (deviation / temp_tolerance))
            else:
                temp_score = 0.0  # Uncomfortable

            score_components.append(temp_score * 0.4)  # 40% weight

            # Humidity comfort score
            humidity = conditions.get('humidity', 50)
            humidity_range = zone_config['humidity_range']
            humidity_tolerance = user_prefs.get('humidity', {}).get(
                'tolerance', 10)

            if humidity_range['min'] - humidity_tolerance <= humidity <= \
                    humidity_range['max'] + humidity_tolerance:
                if humidity_range['min'] <= humidity <= humidity_range['max']:
                    humidity_score = 1.0
                else:
                    deviation = min(abs(humidity - humidity_range['min']),
                                    abs(humidity - humidity_range['max']))
                    humidity_score = max(0, 1.0 - (
                                deviation / humidity_tolerance))
            else:
                humidity_score = 0.0

            score_components.append(humidity_score * 0.2)  # 20% weight

            # Lighting comfort score
            brightness = conditions.get('lighting_brightness', 75)
            preferred_brightness = user_prefs.get('lighting', {}).get(
                'brightness_preference', 75)

            brightness_deviation = abs(brightness - preferred_brightness) / 100
            lighting_score = max(0, 1.0 - brightness_deviation)

            score_components.append(lighting_score * 0.3)  # 30% weight

            # Air quality score (simplified)
            air_quality = conditions.get('air_quality', 'good')
            air_quality_scores = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6,
                                  'poor': 0.3}
            air_score = air_quality_scores.get(air_quality, 0.5)

            score_components.append(air_score * 0.1)  # 10% weight

            # Calculate overall comfort score
            overall_score = sum(score_components)

            return min(1.0, max(0.0, overall_score))

        except Exception as e:
            logger.error(f"Error calculating comfort score: {e}")
            return 0.5  # Neutral score

    def _analyze_comfort_factors(self, zone_id: str,
                                 conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting comfort in a zone"""
        zone_config = self.comfort_zones[zone_id]
        factors = {}

        # Temperature factor
        temp = conditions.get('temperature', 72)
        temp_range = zone_config['temperature_range']

        if temp < temp_range['min']:
            factors['temperature'] = 'too_cold'
        elif temp > temp_range['max']:
            factors['temperature'] = 'too_warm'
        else:
            factors['temperature'] = 'comfortable'

        # Humidity factor
        humidity = conditions.get('humidity', 50)
        humidity_range = zone_config['humidity_range']

        if humidity < humidity_range['min']:
            factors['humidity'] = 'too_dry'
        elif humidity > humidity_range['max']:
            factors['humidity'] = 'too_humid'
        else:
            factors['humidity'] = 'comfortable'

        # Lighting factor
        brightness = conditions.get('lighting_brightness', 75)

        if brightness < 30:
            factors['lighting'] = 'too_dim'
        elif brightness > 90:
            factors['lighting'] = 'too_bright'
        else:
            factors['lighting'] = 'comfortable'

        return factors

    async def _analyze_occupancy_patterns(self):
        """Analyze occupancy patterns to optimize comfort scheduling"""
        try:
            current_time = datetime.utcnow()

            for zone_id in self.comfort_zones:
                occupancy = await self._detect_occupancy(zone_id)

                # Store occupancy pattern
                if zone_id not in self.occupancy_patterns:
                    self.occupancy_patterns[zone_id] = []

                self.occupancy_patterns[zone_id].append({
                    'timestamp': current_time,
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'occupied': occupancy
                })

                # Keep only recent history (last 7 days)
                week_ago = current_time - timedelta(days=7)
                self.occupancy_patterns[zone_id] = [
                    pattern for pattern in self.occupancy_patterns[zone_id]
                    if pattern['timestamp'] > week_ago
                ]

        except Exception as e:
            logger.error(f"Error analyzing occupancy patterns: {e}")

    async def _optimize_comfort_settings(self):
        """Optimize comfort settings based on occupancy and preferences"""
        try:
            for zone_id, zone_config in self.comfort_zones.items():
                conditions = zone_config.get('current_conditions', {})
                occupancy = conditions.get('occupancy', False)

                if occupancy:
                    # Zone is occupied, optimize for comfort
                    await self._optimize_occupied_zone(zone_id, conditions)
                else:
                    # Zone is unoccupied, optimize for energy efficiency
                    await self._optimize_unoccupied_zone(zone_id, conditions)

        except Exception as e:
            logger.error(f"Error optimizing comfort settings: {e}")

    async def _optimize_occupied_zone(self, zone_id: str,
                                      conditions: Dict[str, Any]):
        """Optimize settings for an occupied zone"""
        try:
            zone_config = self.comfort_zones[zone_id]
            comfort_score = await self._calculate_comfort_score(zone_id,
                                                                conditions)

            # If comfort score is low, make adjustments
            if comfort_score < 0.7:
                comfort_factors = self._analyze_comfort_factors(zone_id,
                                                                conditions)

                # Temperature adjustment
                if comfort_factors.get('temperature') == 'too_cold':
                    target_temp = conditions.get('temperature', 72) + 2
                    await self._request_temperature_adjustment(zone_id,
                                                               target_temp)
                elif comfort_factors.get('temperature') == 'too_warm':
                    target_temp = conditions.get('temperature', 72) - 2
                    await self._request_temperature_adjustment(zone_id,
                                                               target_temp)

                # Lighting adjustment
                if comfort_factors.get('lighting') == 'too_dim':
                    target_brightness = min(100, conditions.get(
                        'lighting_brightness', 75) + 20)
                    await self._request_lighting_adjustment(zone_id,
                                                            target_brightness)
                elif comfort_factors.get('lighting') == 'too_bright':
                    target_brightness = max(10, conditions.get(
                        'lighting_brightness', 75) - 20)
                    await self._request_lighting_adjustment(zone_id,
                                                            target_brightness)

                logger.info(
                    f"Optimized comfort settings for occupied zone {zone_id} (score: {comfort_score:.2f})")

        except Exception as e:
            logger.error(f"Error optimizing occupied zone {zone_id}: {e}")

    async def _optimize_unoccupied_zone(self, zone_id: str,
                                        conditions: Dict[str, Any]):
        """Optimize settings for an unoccupied zone (energy efficiency focus)"""
        try:
            # Apply energy-saving measures
            current_temp = conditions.get('temperature', 72)

            # Temperature setback
            if self.comfort_priority != 'comfort_first':
                # Setback temperature when unoccupied
                if current_temp > 75:  # Cooling season
                    setback_temp = current_temp + 3  # Raise cooling setpoint
                    await self._request_temperature_adjustment(zone_id,
                                                               setback_temp)
                elif current_temp < 70:  # Heating season
                    setback_temp = current_temp - 3  # Lower heating setpoint
                    await self._request_temperature_adjustment(zone_id,
                                                               setback_temp)

            # Lighting reduction
            current_brightness = conditions.get('lighting_brightness', 75)
            if current_brightness > 20:
                # Reduce lighting to minimum level
                await self._request_lighting_adjustment(zone_id, 10)

            logger.debug(
                f"Applied energy-saving measures to unoccupied zone {zone_id}")

        except Exception as e:
            logger.error(f"Error optimizing unoccupied zone {zone_id}: {e}")

    async def _request_temperature_adjustment(self, zone_id: str,
                                              target_temp: float):
        """Request temperature adjustment from HVAC agent"""
        try:
            await self.send_message('hvac', 'request', {
                'request_type': 'adjust_temperature',
                'zone': zone_id,
                'temperature': target_temp,
                'requester': 'comfort_optimization',
                'priority': 2  # Medium priority
            })
        except Exception as e:
            logger.error(f"Error requesting temperature adjustment: {e}")

    async def _request_lighting_adjustment(self, zone_id: str,
                                           brightness: int):
        """Request lighting adjustment from lighting agent"""
        try:
            await self.send_message('lighting', 'request', {
                'request_type': 'set_lighting',
                'room': zone_id,
                'brightness': brightness,
                'requester': 'comfort_optimization'
            })
        except Exception as e:
            logger.error(f"Error requesting lighting adjustment: {e}")

    async def _learn_from_user_behavior(self):
        """Learn from user behavior and preferences"""
        try:
            if not self.learning_enabled:
                return

            # Analyze user interventions (manual adjustments)
            interventions = self.comfort_metrics.get('user_interventions', [])

            if len(interventions) >= self.learning_params[
                'minimum_data_points']:
                # Look for patterns in user adjustments
                patterns = await self._identify_user_patterns(interventions)

                if patterns:
                    # Update preferences based on learned patterns
                    await self._update_preferences_from_patterns(patterns)

                    logger.info(
                        f"Learned {len(patterns)} user preference patterns")

        except Exception as e:
            logger.error(f"Error learning from user behavior: {e}")

    async def _identify_user_patterns(self,
                                      interventions: List[Dict[str, Any]]) -> \
    List[Dict[str, Any]]:
        """Identify patterns in user interventions"""
        try:
            patterns = []

            # Group interventions by type and analyze
            temp_adjustments = [i for i in interventions if
                                i.get('type') == 'temperature']
            lighting_adjustments = [i for i in interventions if
                                    i.get('type') == 'lighting']

            # Analyze temperature patterns
            if len(temp_adjustments) >= 5:
                avg_adjustment = sum(adj.get('adjustment', 0) for adj in
                                     temp_adjustments) / len(temp_adjustments)

                if abs(avg_adjustment) > 1:  # Significant pattern
                    patterns.append({
                        'type': 'temperature_preference',
                        'adjustment': avg_adjustment,
                        'confidence': min(1.0, len(temp_adjustments) / 20),
                        'context': self._analyze_adjustment_context(
                            temp_adjustments)
                    })

            # Analyze lighting patterns
            if len(lighting_adjustments) >= 5:
                avg_adjustment = sum(adj.get('adjustment', 0) for adj in
                                     lighting_adjustments) / len(
                    lighting_adjustments)

                if abs(avg_adjustment) > 10:  # Significant pattern (brightness scale 0-100)
                    patterns.append({
                        'type': 'lighting_preference',
                        'adjustment': avg_adjustment,
                        'confidence': min(1.0, len(lighting_adjustments) / 20),
                        'context': self._analyze_adjustment_context(
                            lighting_adjustments)
                    })

            return patterns

        except Exception as e:
            logger.error(f"Error identifying user patterns: {e}")
            return []

    def _analyze_adjustment_context(self, adjustments: List[Dict[str, Any]]) -> \
    Dict[str, Any]:
        """Analyze the context of user adjustments"""
        context = {}

        # Time-based patterns
        hours = [adj.get('hour', 12) for adj in adjustments]
        if hours:
            context['common_hours'] = list(
                set([h for h in hours if hours.count(h) > len(hours) * 0.3]))

        # Zone-based patterns
        zones = [adj.get('zone', 'unknown') for adj in adjustments]
        if zones:
            context['common_zones'] = list(
                set([z for z in zones if zones.count(z) > len(zones) * 0.3]))

        return context

    async def _update_preferences_from_patterns(self, patterns: List[
        Dict[str, Any]]):
        """Update user preferences based on learned patterns"""
        try:
            for pattern in patterns:
                pattern_type = pattern['type']
                adjustment = pattern['adjustment']
                confidence = pattern['confidence']

                if confidence > 0.5:  # Only apply high-confidence patterns
                    if pattern_type == 'temperature_preference':
                        # Adjust temperature preferences
                        current_heating = self.user_preferences.get(
                            'temperature', {}).get('heating_setpoint', 70)
                        current_cooling = self.user_preferences.get(
                            'temperature', {}).get('cooling_setpoint', 76)

                        # Apply learned adjustment
                        new_heating = current_heating + adjustment * 0.5  # Conservative adjustment
                        new_cooling = current_cooling + adjustment * 0.5

                        # Update preferences
                        if 'temperature' not in self.user_preferences:
                            self.user_preferences['temperature'] = {}

                        self.user_preferences['temperature'][
                            'heating_setpoint'] = new_heating
                        self.user_preferences['temperature'][
                            'cooling_setpoint'] = new_cooling

                        logger.info(
                            f"Updated temperature preferences: heating {new_heating:.1f}°F, cooling {new_cooling:.1f}°F")

                    elif pattern_type == 'lighting_preference':
                        # Adjust lighting preferences
                        current_brightness = self.user_preferences.get(
                            'lighting', {}).get('brightness_preference', 75)
                        new_brightness = max(10, min(100,
                                                     current_brightness + adjustment * 0.3))

                        if 'lighting' not in self.user_preferences:
                            self.user_preferences['lighting'] = {}

                        self.user_preferences['lighting'][
                            'brightness_preference'] = new_brightness

                        logger.info(
                            f"Updated lighting preference: brightness {new_brightness:.0f}%")

            # Save updated preferences to database
            await self._save_user_preferences()

        except Exception as e:
            logger.error(f"Error updating preferences from patterns: {e}")

    async def _save_user_preferences(self):
        """Save updated user preferences to database"""
        try:
            # Generate embedding for preferences
            preferences_text = json.dumps(self.user_preferences)
            embedding = await self.embedding_service.generate_embedding(
                preferences_text)

            await self.db_service.store_user_preferences({
                'user_id': 'default_user',
                'preference_type': 'comfort_preferences',
                'preferences': self.user_preferences,
                'preference_embedding': embedding,
                'updated_at': datetime.utcnow(),
                'learning_enabled': self.learning_enabled
            })

            logger.info("Saved updated user preferences")

        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    async def _generate_comfort_recommendations(self):
        """Generate comfort recommendations for users"""
        try:
            recommendations = []

            # Analyze current comfort scores across zones
            low_comfort_zones = []

            for zone_id, zone_config in self.comfort_zones.items():
                conditions = zone_config.get('current_conditions', {})
                if conditions:
                    comfort_score = await self._calculate_comfort_score(
                        zone_id, conditions)

                    if comfort_score < 0.6:  # Low comfort score
                        low_comfort_zones.append({
                            'zone': zone_id,
                            'score': comfort_score,
                            'issues': self._analyze_comfort_factors(zone_id,
                                                                    conditions)
                        })

            # Generate recommendations for low comfort zones
            for zone_info in low_comfort_zones:
                zone_id = zone_info['zone']
                issues = zone_info['issues']

                zone_recommendations = []

                if issues.get('temperature') == 'too_cold':
                    zone_recommendations.append(
                        "Consider increasing the heating setpoint")
                elif issues.get('temperature') == 'too_warm':
                    zone_recommendations.append(
                        "Consider decreasing the cooling setpoint")

                if issues.get('humidity') == 'too_dry':
                    zone_recommendations.append("Consider using a humidifier")
                elif issues.get('humidity') == 'too_humid':
                    zone_recommendations.append(
                        "Consider improving ventilation or using a dehumidifier")

                if issues.get('lighting') == 'too_dim':
                    zone_recommendations.append(
                        "Consider increasing lighting brightness")
                elif issues.get('lighting') == 'too_bright':
                    zone_recommendations.append("Consider dimming the lights")

                if zone_recommendations:
                    recommendations.append({
                        'zone': zone_id,
                        'comfort_score': zone_info['score'],
                        'recommendations': zone_recommendations,
                        'priority': 'high' if zone_info[
                                                  'score'] < 0.4 else 'medium'
                    })

            # Energy efficiency recommendations
            if self.comfort_priority != 'comfort_first':
                energy_recommendations = await self._generate_energy_efficiency_recommendations()
                recommendations.extend(energy_recommendations)

            # Broadcast recommendations if any
            if recommendations:
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'comfort_recommendations',
                        'recommendations': recommendations,
                        'generated_at': datetime.utcnow().isoformat()
                    }
                )

        except Exception as e:
            logger.error(f"Error generating comfort recommendations: {e}")

    async def _generate_energy_efficiency_recommendations(self) -> List[
        Dict[str, Any]]:
        """Generate energy efficiency recommendations that maintain comfort"""
        try:
            recommendations = []

            # Check for unoccupied zones with high energy usage
            for zone_id, zone_config in self.comfort_zones.items():
                conditions = zone_config.get('current_conditions', {})
                occupancy = conditions.get('occupancy', False)

                if not occupancy:
                    # Zone is unoccupied, check for energy waste
                    temp = conditions.get('temperature', 72)
                    brightness = conditions.get('lighting_brightness', 0)

                    if brightness > 50:
                        recommendations.append({
                            'zone': zone_id,
                            'type': 'energy_efficiency',
                            'recommendation': f"Reduce lighting in unoccupied {zone_id}",
                            'potential_savings': '10-15%',
                            'comfort_impact': 'none'
                        })

                    if temp < 65 or temp > 78:  # Extreme temperatures when unoccupied
                        recommendations.append({
                            'zone': zone_id,
                            'type': 'energy_efficiency',
                            'recommendation': f"Adjust temperature setback in unoccupied {zone_id}",
                            'potential_savings': '15-20%',
                            'comfort_impact': 'none'
                        })

            return recommendations

        except Exception as e:
            logger.error(
                f"Error generating energy efficiency recommendations: {e}")
            return []

    async def _resolve_comfort_conflicts(self):
        """Resolve conflicts between comfort and energy efficiency"""
        try:
            # This method would handle situations where comfort optimization
            # conflicts with energy efficiency goals

            # Get current energy emergency status
            try:
                energy_response = await self._request_from_agent(
                    'energy_monitor', 'get_consumption_data', {'hours': 1}
                )
                high_consumption = len(
                    energy_response.get('consumption_data', [])) > 0
            except:
                high_consumption = False

            if high_consumption and self.comfort_priority == 'balanced':
                # Temporarily prioritize energy efficiency
                logger.info(
                    "High energy consumption detected, temporarily prioritizing efficiency")

                # Apply more aggressive energy-saving measures
                for zone_id in self.comfort_zones:
                    conditions = self.comfort_zones[zone_id].get(
                        'current_conditions', {})
                    occupancy = conditions.get('occupancy', False)

                    if not occupancy:
                        # More aggressive setbacks for unoccupied zones
                        current_temp = conditions.get('temperature', 72)

                        if current_temp > 74:
                            await self._request_temperature_adjustment(zone_id,
                                                                       current_temp + 4)
                        elif current_temp < 68:
                            await self._request_temperature_adjustment(zone_id,
                                                                       current_temp - 4)

                        # Turn off lighting
                        await self._request_lighting_adjustment(zone_id, 0)

        except Exception as e:
            logger.error(f"Error resolving comfort conflicts: {e}")

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

    async def _handle_occupancy_change(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle occupancy change notifications"""
        zone_id = data.get('room', data.get('zone', 'unknown'))
        occupied = data.get('occupied', False)

        if zone_id in self.comfort_zones:
            logger.info(
                f"Occupancy change in {zone_id}: {'occupied' if occupied else 'unoccupied'}")

            # Update zone conditions
            if 'current_conditions' not in self.comfort_zones[zone_id]:
                self.comfort_zones[zone_id]['current_conditions'] = {}

            self.comfort_zones[zone_id]['current_conditions'][
                'occupancy'] = occupied

            # Trigger immediate optimization for this zone
            conditions = self.comfort_zones[zone_id]['current_conditions']

            if occupied:
                await self._optimize_occupied_zone(zone_id, conditions)
            else:
                await self._optimize_unoccupied_zone(zone_id, conditions)

    async def _handle_user_feedback(self, sender: str, data: Dict[str, Any]):
        """Handle user feedback about comfort"""
        feedback_type = data.get('feedback_type', 'comfort')
        zone_id = data.get('zone', 'unknown')
        rating = data.get('rating', 5)  # 1-10 scale
        comments = data.get('comments', '')

        # Store feedback for learning
        feedback_record = {
            'timestamp': datetime.utcnow(),
            'zone': zone_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'comments': comments,
            'conditions_at_feedback': self.comfort_zones.get(zone_id, {}).get(
                'current_conditions', {})
        }

        self.comfort_metrics['user_interventions'].append(feedback_record)

        logger.info(
            f"Received user feedback for {zone_id}: {rating}/10 - {comments}")

        # Immediate adjustment if feedback is very negative
        if rating <= 3:
            conditions = self.comfort_zones.get(zone_id, {}).get(
                'current_conditions', {})
            if conditions:
                # Make immediate comfort improvements
                await self._optimize_occupied_zone(zone_id, conditions)

    async def _handle_energy_emergency(self, sender: str,
                                       data: Dict[str, Any]):
        """Handle energy emergency events"""
        severity = data.get('severity', 'medium')

        if severity == 'critical':
            # Temporarily override comfort settings for energy savings
            self.comfort_priority = 'efficiency_first'

            logger.warning(
                "Energy emergency: switching to efficiency-first mode")

            # Apply immediate energy-saving measures
            for zone_id in self.comfort_zones:
                conditions = self.comfort_zones[zone_id].get(
                    'current_conditions', {})
                occupancy = conditions.get('occupancy', False)

                if not occupancy:
                    # Aggressive setbacks for unoccupied zones
                    await self._request_temperature_adjustment(zone_id,
                                                               78 if conditions.get(
                                                                   'temperature',
                                                                   72) > 72 else 65)
                    await self._request_lighting_adjustment(zone_id, 0)
                else:
                    # Moderate setbacks for occupied zones
                    current_temp = conditions.get('temperature', 72)
                    await self._request_temperature_adjustment(zone_id,
                                                               current_temp + 2)

        elif severity == 'medium':
            # Moderate energy-saving measures
            self.comfort_priority = 'balanced'

            # Apply comfort-conscious energy savings
            for zone_id in self.comfort_zones:
                conditions = self.comfort_zones[zone_id].get(
                    'current_conditions', {})
                occupancy = conditions.get('occupancy', False)

                if not occupancy:
                    await self._optimize_unoccupied_zone(zone_id, conditions)

    async def _handle_comfort_preferences_request(self, sender: str,
                                                  data: Dict[str, Any]):
        """Handle request for comfort preferences"""
        zone_id = data.get('zone')

        if zone_id and zone_id in self.comfort_zones:
            zone_preferences = {
                'zone_config': self.comfort_zones[zone_id],
                'user_preferences': self.user_preferences,
                'current_conditions': self.comfort_zones[zone_id].get(
                    'current_conditions', {})
            }
        else:
            zone_preferences = {
                'user_preferences': self.user_preferences,
                'comfort_zones': self.comfort_zones
            }

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'comfort_preferences': zone_preferences,
            'comfort_priority': self.comfort_priority,
            'learning_enabled': self.learning_enabled
        })

    async def _handle_set_preferences_request(self, sender: str,
                                              data: Dict[str, Any]):
        """Handle request to set comfort preferences"""
        new_preferences = data.get('preferences', {})

        # Update preferences
        for key, value in new_preferences.items():
            if key in self.user_preferences:
                if isinstance(value, dict) and isinstance(
                        self.user_preferences[key], dict):
                    self.user_preferences[key].update(value)
                else:
                    self.user_preferences[key] = value
            else:
                self.user_preferences[key] = value

        # Save updated preferences
        await self._save_user_preferences()

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'status': 'success',
            'updated_preferences': self.user_preferences
        })

        logger.info("Updated user comfort preferences")

    async def _handle_set_priority_request(self, sender: str,
                                           data: Dict[str, Any]):
        """Handle request to set comfort priority"""
        new_priority = data.get('priority', 'balanced')

        if new_priority in ['comfort_first', 'balanced', 'efficiency_first']:
            self.comfort_priority = new_priority

            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'success',
                'comfort_priority': self.comfort_priority
            })

            logger.info(f"Updated comfort priority to: {new_priority}")
        else:
            await self.send_message(sender, 'response', {
                'request_id': data.get('request_id'),
                'status': 'error',
                'message': 'Invalid priority value'
            })