import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

from core.base_agent import BaseAgent
from services.pricing_service import PricingService
from config.settings import settings

logger = logging.getLogger(__name__)


class PriceIntelligenceAgent(BaseAgent):
    """Agent responsible for monitoring energy prices and optimizing consumption timing"""

    def __init__(self):
        super().__init__("price_intelligence", "PriceIntelligence")
        self.pricing_service = PricingService()
        self.current_price_data = {}
        self.price_predictions = {}
        self.peak_hours = []
        self.off_peak_hours = []
        self.price_history = []
        self.optimization_recommendations = []

    async def initialize(self):
        """Initialize the price intelligence agent"""
        logger.info("Initializing Price Intelligence Agent")

        # Subscribe to pricing events
        await self.message_broker.subscribe_to_events(
            self.agent_id,
            ['price_update', 'tariff_change', 'demand_response_event']
        )

        # Load current pricing data
        await self._load_current_pricing()

        # Initialize price predictions
        await self._initialize_price_predictions()

        # Set up peak/off-peak hours
        await self._configure_time_periods()

        logger.info("Price Intelligence Agent initialized successfully")

    async def execute(self):
        """Main execution logic for price intelligence"""
        try:
            # Update current pricing data
            await self._update_pricing_data()

            # Generate price predictions
            await self._generate_price_predictions()

            # Analyze pricing trends
            await self._analyze_pricing_trends()

            # Generate optimization recommendations
            await self._generate_optimization_recommendations()

            # Monitor for demand response events
            await self._monitor_demand_response()

            # Notify other agents of price changes
            await self._notify_price_changes()

        except Exception as e:
            logger.error(f"Error in price intelligence execution: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Price Intelligence Agent")

    async def handle_request(self, sender: str, data: Dict[str, Any]):
        """Handle requests from other agents"""
        request_type = data.get('request_type')

        if request_type == 'get_current_price':
            await self._handle_current_price_request(sender, data)
        elif request_type == 'get_price_forecast':
            await self._handle_price_forecast_request(sender, data)
        elif request_type == 'get_optimal_schedule':
            await self._handle_optimal_schedule_request(sender, data)
        elif request_type == 'check_peak_hours':
            await self._handle_peak_hours_request(sender, data)
        elif request_type == 'calculate_cost_savings':
            await self._handle_cost_savings_request(sender, data)
        else:
            logger.warning(f"Unknown request type: {request_type}")

    async def handle_notification(self, sender: str, data: Dict[str, Any]):
        """Handle notifications from other agents"""
        notification_type = data.get('notification_type')

        if notification_type == 'consumption_planned':
            await self._handle_consumption_notification(sender, data)
        elif notification_type == 'device_schedule_change':
            await self._handle_device_schedule_change(sender, data)
        elif notification_type == 'emergency_event':
            await self._handle_emergency_event(sender, data)

    async def _load_current_pricing(self):
        """Load current pricing information"""
        try:
            # Get current pricing from service
            pricing_data = await self.pricing_service.get_current_pricing()

            if pricing_data:
                self.current_price_data = pricing_data

                # Store in database with embedding
                pricing_embedding = await self.embedding_service.embed_price_data(
                    pricing_data)
                pricing_data['price_embedding'] = pricing_embedding

                await self.db_service.store_price_data(pricing_data)

                logger.info(
                    f"Loaded current pricing: ${pricing_data.get('price_per_kwh', 0)}/kWh")
            else:
                logger.warning("Failed to load current pricing data")

        except Exception as e:
            logger.error(f"Error loading current pricing: {e}")

    async def _initialize_price_predictions(self):
        """Initialize price prediction models"""
        try:
            # Get historical price data for the last 30 days
            historical_data = []

            # Query historical pricing from database
            query = {
                "range": {
                    "timestamp": {
                        "gte": "now-30d"
                    }
                }
            }

            price_history = await self.db_service.search_documents(
                "price_data", query, size=1000)

            if price_history:
                self.price_history = price_history
                logger.info(
                    f"Loaded {len(price_history)} historical price records")
            else:
                logger.warning("No historical price data available")

        except Exception as e:
            logger.error(f"Error initializing price predictions: {e}")

    async def _configure_time_periods(self):
        """Configure peak and off-peak time periods"""
        try:
            # Default time periods (can be customized based on utility company)
            self.peak_hours = [16, 17, 18, 19, 20, 21]  # 4 PM - 9 PM
            self.off_peak_hours = [22, 23, 0, 1, 2, 3, 4, 5]  # 10 PM - 5 AM

            # Could be enhanced to load from utility company API or configuration
            logger.info(f"Configured peak hours: {self.peak_hours}")
            logger.info(f"Configured off-peak hours: {self.off_peak_hours}")

        except Exception as e:
            logger.error(f"Error configuring time periods: {e}")

    async def _update_pricing_data(self):
        """Update current pricing data"""
        try:
            # Get latest pricing from service
            new_pricing = await self.pricing_service.get_current_pricing()

            if new_pricing and new_pricing != self.current_price_data:
                old_price = self.current_price_data.get('price_per_kwh', 0)
                new_price = new_pricing.get('price_per_kwh', 0)

                # Store previous price data
                if self.current_price_data:
                    self.price_history.append(self.current_price_data)

                    # Keep only recent history (last 1000 records)
                    if len(self.price_history) > 1000:
                        self.price_history = self.price_history[-1000:]

                # Update current price
                self.current_price_data = new_pricing

                # Generate embedding and store
                pricing_embedding = await self.embedding_service.embed_price_data(
                    new_pricing)
                new_pricing['price_embedding'] = pricing_embedding

                await self.db_service.store_price_data(new_pricing)

                # Log price change
                price_change = ((
                                            new_price - old_price) / old_price * 100) if old_price > 0 else 0
                logger.info(
                    f"Price updated: ${new_price}/kWh ({price_change:+.1f}%)")

                # Notify about significant price changes
                if abs(price_change) > 10:  # More than 10% change
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'significant_price_change',
                            'old_price': old_price,
                            'new_price': new_price,
                            'change_percentage': price_change,
                            'severity': 'high' if abs(
                                price_change) > 25 else 'medium'
                        }
                    )

        except Exception as e:
            logger.error(f"Error updating pricing data: {e}")

    async def _generate_price_predictions(self):
        """Generate price predictions for the next 24 hours"""
        try:
            if not self.price_history:
                logger.warning("No price history available for predictions")
                return

            # Prepare historical context for LLM
            recent_prices = self.price_history[-48:] if len(
                self.price_history) >= 48 else self.price_history

            historical_context = {
                'current_price': self.current_price_data.get('price_per_kwh',
                                                             0),
                'recent_trend': self._calculate_price_trend(recent_prices),
                'average_price_24h': np.mean(
                    [p.get('price_per_kwh', 0) for p in recent_prices[-24:]]),
                'price_volatility': np.std(
                    [p.get('price_per_kwh', 0) for p in recent_prices]),
                'current_hour': datetime.utcnow().hour,
                'day_of_week': datetime.utcnow().strftime('%A'),
                'season': self._get_current_season()
            }

            # External factors affecting pricing
            external_factors = {
                'demand_forecast': await self._get_demand_forecast(),
                'weather_impact': await self._get_weather_impact(),
                'grid_events': await self._get_grid_events(),
                'market_conditions': 'normal'
                # Could be enhanced with real market data
            }

            # Generate predictions using LLM
            predictions = await self.llm_client.predict_energy_demand(
                historical_context,
                external_factors
            )

            if predictions and 'hourly_predictions' in predictions:
                self.price_predictions = predictions

                # Store predictions in database
                prediction_record = {
                    'timestamp': datetime.utcnow(),
                    'prediction_type': 'price_forecast',
                    'predictions': predictions,
                    'confidence': predictions.get('confidence', 0.5)
                }

                await self.db_service.store_document('price_data',
                                                     prediction_record)

                logger.info(f"Generated price predictions for next 24 hours")

        except Exception as e:
            logger.error(f"Error generating price predictions: {e}")

    async def _analyze_pricing_trends(self):
        """Analyze pricing trends and patterns"""
        try:
            if len(self.price_history) < 24:
                return

            # Analyze recent pricing patterns
            recent_prices = [p.get('price_per_kwh', 0) for p in
                             self.price_history[-168:]]  # Last 7 days

            trend_analysis = {
                'current_trend': self._calculate_price_trend(
                    self.price_history[-24:]),
                'weekly_average': np.mean(recent_prices),
                'price_volatility': np.std(recent_prices),
                'peak_hour_premium': self._calculate_peak_premium(),
                'savings_potential': self._calculate_savings_potential()
            }

            # Generate insights using LLM
            analysis_context = {
                'trend_data': trend_analysis,
                'historical_prices': self.price_history[-48:],
                'current_price': self.current_price_data
            }

            # Store trend analysis
            trend_record = {
                'timestamp': datetime.utcnow(),
                'analysis_type': 'pricing_trends',
                'trend_analysis': trend_analysis,
                'insights': 'Generated by price intelligence agent'
            }

            await self.db_service.store_document('price_data', trend_record)

        except Exception as e:
            logger.error(f"Error analyzing pricing trends: {e}")

    async def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on pricing"""
        try:
            current_hour = datetime.utcnow().hour
            current_price = self.current_price_data.get('price_per_kwh', 0)

            recommendations = []

            # Peak hour recommendations
            if current_hour in self.peak_hours:
                recommendations.append({
                    'type': 'peak_hour_optimization',
                    'message': 'Currently in peak pricing period. Consider delaying non-essential device usage.',
                    'priority': 'high',
                    'potential_savings': 20,
                    'recommended_actions': [
                        'Delay dishwasher and laundry cycles',
                        'Reduce HVAC usage if possible',
                        'Use battery storage if available'
                    ]
                })

            # Off-peak recommendations
            elif current_hour in self.off_peak_hours:
                recommendations.append({
                    'type': 'off_peak_optimization',
                    'message': 'Currently in off-peak pricing period. Good time for energy-intensive tasks.',
                    'priority': 'medium',
                    'potential_savings': 15,
                    'recommended_actions': [
                        'Run dishwasher and laundry',
                        'Charge electric vehicles',
                        'Pre-heat/cool home for comfort'
                    ]
                })

            # Price prediction based recommendations
            if self.price_predictions and 'hourly_predictions' in self.price_predictions:
                next_6_hours = self.price_predictions['hourly_predictions'][:6]

                # Find lowest price period in next 6 hours
                if next_6_hours:
                    min_price_hour = min(next_6_hours, key=lambda x: x.get(
                        'predicted_consumption', float('inf')))

                    if min_price_hour[
                        'predicted_consumption'] < current_price * 0.8:  # 20% cheaper
                        recommendations.append({
                            'type': 'timing_optimization',
                            'message': f"Lower prices expected at {min_price_hour['hour']}:00. Consider scheduling energy use then.",
                            'priority': 'medium',
                            'optimal_time': f"{min_price_hour['hour']}:00",
                            'potential_savings': 20
                        })

            self.optimization_recommendations = recommendations

            # Broadcast recommendations to other agents
            if recommendations:
                await self.broadcast_message(
                    'notification',
                    {
                        'notification_type': 'pricing_recommendations',
                        'recommendations': recommendations,
                        'current_price': current_price,
                        'price_tier': self._get_current_price_tier()
                    }
                )

        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")

    async def _monitor_demand_response(self):
        """Monitor for demand response events"""
        try:
            # Check for demand response events from utility
            dr_events = await self.pricing_service.get_demand_response_events()

            if dr_events:
                for event in dr_events:
                    event_type = event.get('event_type')
                    start_time = event.get('start_time')
                    end_time = event.get('end_time')
                    incentive = event.get('incentive', 0)

                    # Notify all agents about demand response event
                    await self.broadcast_message(
                        'notification',
                        {
                            'notification_type': 'demand_response_event',
                            'event_type': event_type,
                            'start_time': start_time,
                            'end_time': end_time,
                            'incentive': incentive,
                            'recommended_reduction': event.get(
                                'target_reduction', 0)
                        }
                    )

                    logger.info(
                        f"Demand response event: {event_type} from {start_time} to {end_time}")

        except Exception as e:
            logger.error(f"Error monitoring demand response: {e}")

    async def _notify_price_changes(self):
        """Notify other agents about price changes"""
        try:
            if not self.current_price_data:
                return

            current_price = self.current_price_data.get('price_per_kwh', 0)
            price_tier = self._get_current_price_tier()

            # Periodic price notification (every update cycle)
            await self.broadcast_message(
                'notification',
                {
                    'notification_type': 'price_update',
                    'current_price': current_price,
                    'price_tier': price_tier,
                    'timestamp': datetime.utcnow().isoformat(),
                    'peak_hours': self.peak_hours,
                    'off_peak_hours': self.off_peak_hours
                }
            )

        except Exception as e:
            logger.error(f"Error notifying price changes: {e}")

    def _calculate_price_trend(self, price_data: List[Dict[str, Any]]) -> str:
        """Calculate price trend from historical data"""
        if len(price_data) < 2:
            return 'stable'

        prices = [p.get('price_per_kwh', 0) for p in price_data]

        # Calculate moving average trend
        if len(prices) >= 6:
            recent_avg = np.mean(prices[-3:])  # Last 3 periods
            earlier_avg = np.mean(prices[-6:-3])  # Previous 3 periods

            if recent_avg > earlier_avg * 1.05:  # 5% increase
                return 'increasing'
            elif recent_avg < earlier_avg * 0.95:  # 5% decrease
                return 'decreasing'

        return 'stable'

    def _get_current_season(self) -> str:
        """Get current season"""
        month = datetime.utcnow().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def _get_current_price_tier(self) -> str:
        """Get current price tier"""
        current_hour = datetime.utcnow().hour

        if current_hour in self.peak_hours:
            return 'peak'
        elif current_hour in self.off_peak_hours:
            return 'off_peak'
        else:
            return 'standard'

    def _calculate_peak_premium(self) -> float:
        """Calculate peak hour price premium"""
        if not self.price_history:
            return 0.0

        peak_prices = []
        off_peak_prices = []

        for price_data in self.price_history[-168:]:  # Last week
            timestamp = price_data.get('timestamp')
            if timestamp:
                hour = datetime.fromisoformat(timestamp).hour
                price = price_data.get('price_per_kwh', 0)

                if hour in self.peak_hours:
                    peak_prices.append(price)
                elif hour in self.off_peak_hours:
                    off_peak_prices.append(price)

        if peak_prices and off_peak_prices:
            avg_peak = np.mean(peak_prices)
            avg_off_peak = np.mean(off_peak_prices)

            if avg_off_peak > 0:
                return ((avg_peak - avg_off_peak) / avg_off_peak) * 100

        return 0.0

    def _calculate_savings_potential(self) -> float:
        """Calculate potential savings from time shifting"""
        peak_premium = self._calculate_peak_premium()

        # Estimate savings potential based on peak premium and typical usage patterns
        if peak_premium > 50:  # High peak premium
            return 25.0  # Up to 25% savings potential
        elif peak_premium > 25:
            return 15.0  # Up to 15% savings potential
        else:
            return 5.0  # Up to 5% savings potential

    async def _get_demand_forecast(self) -> Dict[str, Any]:
        """Get demand forecast data"""
        # Placeholder - could integrate with utility demand forecast API
        return {
            'expected_demand': 'normal',
            'peak_probability': 0.3,
            'forecast_confidence': 0.7
        }

    async def _get_weather_impact(self) -> Dict[str, Any]:
        """Get weather impact on pricing"""
        # Placeholder - could integrate with weather service
        return {
            'temperature_impact': 'moderate',
            'cooling_demand': 'normal',
            'heating_demand': 'low'
        }

    async def _get_grid_events(self) -> List[Dict[str, Any]]:
        """Get grid events that might affect pricing"""
        # Placeholder - could integrate with grid operator API
        return []

    async def _handle_current_price_request(self, sender: str,
                                            data: Dict[str, Any]):
        """Handle request for current price information"""
        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'current_price': self.current_price_data,
            'price_tier': self._get_current_price_tier(),
            'timestamp': datetime.utcnow().isoformat()
        })

    async def _handle_price_forecast_request(self, sender: str,
                                             data: Dict[str, Any]):
        """Handle request for price forecast"""
        hours = data.get('hours', 24)

        forecast_data = {}
        if self.price_predictions and 'hourly_predictions' in self.price_predictions:
            forecast_data = {
                'predictions': self.price_predictions['hourly_predictions'][
                               :hours],
                'confidence': self.price_predictions.get('confidence', 0.5)
            }

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'price_forecast': forecast_data,
            'hours_requested': hours
        })

    async def _handle_optimal_schedule_request(self, sender: str,
                                               data: Dict[str, Any]):
        """Handle request for optimal scheduling recommendations"""
        device_type = data.get('device_type')
        energy_requirement = data.get('energy_requirement', 1.0)  # kWh

        optimal_times = []

        if self.price_predictions and 'hourly_predictions' in self.price_predictions:
            # Find hours with lowest predicted prices
            predictions = self.price_predictions['hourly_predictions']
            sorted_hours = sorted(predictions,
                                  key=lambda x: x.get('predicted_consumption',
                                                      float('inf')))

            # Select top 3 optimal hours
            for hour_data in sorted_hours[:3]:
                optimal_times.append({
                    'hour': hour_data['hour'],
                    'predicted_price': hour_data.get('predicted_consumption',
                                                     0),
                    'savings_estimate': self._calculate_hour_savings(
                        hour_data),
                    'confidence': hour_data.get('confidence', 0.5)
                })

        await self.send_message(sender, 'response', {
            'request_id': data.get('request_id'),
            'optimal_schedule': optimal_times,
            'device_type': device_type,
            'energy_requirement': energy_requirement
        })

    def _calculate_hour_savings(self, hour_data: Dict[str, Any]) -> float:
        """Calculate savings for a specific hour"""
        predicted_price = hour_data.get('predicted_consumption', 0)
        current_price = self.current_price_data.get('price_per_kwh', 0)

        if current_price > 0:
            return ((current_price - predicted_price) / current_price) * 100

        return 0.0