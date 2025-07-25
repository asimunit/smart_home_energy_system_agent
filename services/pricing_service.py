import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import json

logger = logging.getLogger(__name__)


class PricingService:
    """Service for energy pricing information and forecasting"""

    def __init__(self):
        self.utility_company = "Northern India Grid"
        self.pricing_structure = "time_of_use"  # flat, time_of_use, real_time
        self.base_rate = 0.12  # Base rate in $/kWh
        self.current_pricing = {}
        self.demand_response_events = []
        self.last_update = None
        self.price_tiers = {
            'off_peak': 0.08,  # 22:00 - 06:00
            'standard': 0.12,  # 06:00 - 16:00, 21:00 - 22:00
            'peak': 0.18  # 16:00 - 21:00
        }

        # Initialize with current pricing
        self._initialize_pricing()
        logger.info("Pricing service initialized")

    def _initialize_pricing(self):
        """Initialize with current pricing data"""
        current_time = datetime.utcnow()
        current_hour = current_time.hour

        # Determine current price tier based on time
        if 22 <= current_hour or current_hour < 6:  # 10 PM - 6 AM
            tier = 'off_peak'
        elif 16 <= current_hour < 21:  # 4 PM - 9 PM
            tier = 'peak'
        else:
            tier = 'standard'

        base_price = self.price_tiers[tier]

        # Add some realistic variation (±5%)
        price_variation = random.uniform(0.95, 1.05)
        current_price = base_price * price_variation

        self.current_pricing = {
            'price_per_kwh': round(current_price, 4),
            'price_tier': tier,
            'utility_company': self.utility_company,
            'pricing_structure': self.pricing_structure,
            'timestamp': current_time.isoformat(),
            'currency': 'USD',
            'taxes_included': True,
            'demand_charge': self._calculate_demand_charge(),
            'transmission_charge': 0.015,  # Fixed transmission charge
            'distribution_charge': 0.025  # Fixed distribution charge
        }

        self.last_update = current_time
        logger.info(f"Initialized pricing: ${current_price:.4f}/kWh ({tier})")

    def _calculate_demand_charge(self) -> float:
        """Calculate demand charge based on time and season"""
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        month = current_time.month

        # Base demand charge varies by season
        if month in [12, 1, 2]:  # Winter
            base_demand = 8.50  # $/kW
        elif month in [6, 7, 8]:  # Summer (peak AC usage)
            base_demand = 15.75  # $/kW
        else:
            base_demand = 12.25  # $/kW

        # Peak hour multiplier
        if 16 <= current_hour < 21:  # Peak hours
            return base_demand * 1.5
        else:
            return base_demand

    async def get_current_pricing(self) -> Dict[str, Any]:
        """Get current energy pricing"""
        try:
            current_time = datetime.utcnow()

            # Update pricing if it's been more than 5 minutes
            if (not self.last_update or
                    (current_time - self.last_update).seconds > 300):
                await self._update_pricing()

            return self.current_pricing.copy()

        except Exception as e:
            logger.error(f"Error getting current pricing: {e}")
            return {}

    async def get_pricing_forecast(self, hours: int = 24) -> List[
        Dict[str, Any]]:
        """Get pricing forecast for specified hours"""
        try:
            forecast = []
            current_time = datetime.utcnow()

            for hour in range(hours):
                forecast_time = current_time + timedelta(hours=hour)
                forecast_hour = forecast_time.hour

                # Determine price tier for this hour
                if 22 <= forecast_hour or forecast_hour < 6:
                    tier = 'off_peak'
                elif 16 <= forecast_hour < 21:
                    tier = 'peak'
                else:
                    tier = 'standard'

                base_price = self.price_tiers[tier]

                # Add day-to-day variation and market factors
                daily_variation = self._get_daily_price_variation(
                    forecast_time)
                market_factor = self._get_market_factor(forecast_time)

                forecast_price = base_price * daily_variation * market_factor

                forecast_entry = {
                    'datetime': forecast_time.isoformat(),
                    'price_per_kwh': round(forecast_price, 4),
                    'price_tier': tier,
                    'confidence': self._calculate_forecast_confidence(hour),
                    'demand_charge': self._calculate_demand_charge_forecast(
                        forecast_time),
                    'factors': {
                        'base_price': base_price,
                        'daily_variation': daily_variation,
                        'market_factor': market_factor
                    }
                }

                forecast.append(forecast_entry)

            return forecast

        except Exception as e:
            logger.error(f"Error getting pricing forecast: {e}")
            return []

    def _get_daily_price_variation(self, forecast_time: datetime) -> float:
        """Get daily price variation factor"""
        day_of_week = forecast_time.weekday()  # 0 = Monday, 6 = Sunday

        # Weekend typically has lower demand
        if day_of_week >= 5:  # Weekend
            return random.uniform(0.90, 1.05)
        else:  # Weekday
            return random.uniform(0.95, 1.15)

    def _get_market_factor(self, forecast_time: datetime) -> float:
        """Get market-based pricing factor"""
        month = forecast_time.month

        # Seasonal market factors
        seasonal_factors = {
            1: 1.10,  # January - winter heating
            2: 1.05,  # February
            3: 0.95,  # March - mild weather
            4: 0.90,  # April
            5: 1.00,  # May
            6: 1.20,  # June - AC season starts
            7: 1.25,  # July - peak summer
            8: 1.22,  # August - peak summer
            9: 1.10,  # September
            10: 0.95,  # October - mild weather
            11: 1.00,  # November
            12: 1.15  # December - winter heating
        }

        base_factor = seasonal_factors.get(month, 1.0)

        # Add some random market volatility (±10%)
        volatility = random.uniform(0.90, 1.10)

        return base_factor * volatility

    def _calculate_forecast_confidence(self, hours_ahead: int) -> float:
        """Calculate confidence level for forecast"""
        # Confidence decreases with time
        if hours_ahead <= 6:
            return 0.95
        elif hours_ahead <= 12:
            return 0.85
        elif hours_ahead <= 24:
            return 0.75
        else:
            return max(0.5, 0.75 - (hours_ahead - 24) * 0.02)

    def _calculate_demand_charge_forecast(self,
                                          forecast_time: datetime) -> float:
        """Calculate forecasted demand charge"""
        hour = forecast_time.hour
        month = forecast_time.month

        # Base seasonal demand charge
        if month in [12, 1, 2]:  # Winter
            base_demand = 8.50
        elif month in [6, 7, 8]:  # Summer
            base_demand = 15.75
        else:
            base_demand = 12.25

        # Peak hour multiplier
        if 16 <= hour < 21:
            return base_demand * 1.5
        else:
            return base_demand

    async def get_demand_response_events(self) -> List[Dict[str, Any]]:
        """Get active demand response events"""
        try:
            # Update demand response events
            await self._update_demand_response_events()

            return self.demand_response_events.copy()

        except Exception as e:
            logger.error(f"Error getting demand response events: {e}")
            return []

    async def _update_demand_response_events(self):
        """Update demand response events"""
        try:
            current_time = datetime.utcnow()

            # Remove expired events
            self.demand_response_events = [
                event for event in self.demand_response_events
                if datetime.fromisoformat(
                    event.get('end_time', '')) > current_time
            ]

            # Simulate new demand response events (5% chance each update)
            if random.random() < 0.05:
                # Generate a new demand response event
                event_types = ['peak_demand_reduction', 'emergency_reduction',
                               'voluntary_reduction']
                event_type = random.choice(event_types)

                start_time = current_time + timedelta(
                    minutes=random.randint(15, 120))
                duration_hours = random.randint(1, 4)
                end_time = start_time + timedelta(hours=duration_hours)

                # Calculate incentive based on event type
                incentives = {
                    'peak_demand_reduction': random.uniform(0.05, 0.15),
                    # $0.05-0.15/kWh
                    'emergency_reduction': random.uniform(0.20, 0.50),
                    # $0.20-0.50/kWh
                    'voluntary_reduction': random.uniform(0.02, 0.08)
                    # $0.02-0.08/kWh
                }

                target_reductions = {
                    'peak_demand_reduction': random.randint(15, 30),  # 15-30%
                    'emergency_reduction': random.randint(30, 50),  # 30-50%
                    'voluntary_reduction': random.randint(10, 20)  # 10-20%
                }

                new_event = {
                    'event_id': f"DR_{int(current_time.timestamp())}",
                    'event_type': event_type,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'incentive': incentives[event_type],
                    'target_reduction': target_reductions[event_type],
                    'utility_company': self.utility_company,
                    'mandatory': event_type == 'emergency_reduction',
                    'description': self._get_event_description(event_type),
                    'created_at': current_time.isoformat()
                }

                self.demand_response_events.append(new_event)
                logger.info(
                    f"New demand response event: {event_type} from {start_time} to {end_time}")

        except Exception as e:
            logger.error(f"Error updating demand response events: {e}")

    def _get_event_description(self, event_type: str) -> str:
        """Get description for demand response event"""
        descriptions = {
            'peak_demand_reduction': "Voluntary demand reduction requested during peak usage period",
            'emergency_reduction': "Mandatory demand reduction due to grid emergency conditions",
            'voluntary_reduction': "Optional demand reduction program with incentive payments"
        }

        return descriptions.get(event_type, "Demand response event")

    async def get_real_time_pricing(self) -> Dict[str, Any]:
        """Get real-time pricing data (if available)"""
        try:
            current_pricing = await self.get_current_pricing()

            # Add real-time specific data
            real_time_data = current_pricing.copy()
            real_time_data.update({
                'pricing_type': 'real_time',
                'market_price': current_pricing['price_per_kwh'] * 0.7,
                # Wholesale market price
                'congestion_charge': random.uniform(0.005, 0.025),
                'line_loss_factor': 1.05,
                'next_update': (datetime.utcnow() + timedelta(
                    minutes=5)).isoformat(),
                'price_trend': self._calculate_price_trend()
            })

            return real_time_data

        except Exception as e:
            logger.error(f"Error getting real-time pricing: {e}")
            return {}

    def _calculate_price_trend(self) -> str:
        """Calculate current price trend"""
        # Simulate price trend based on time of day and random factors
        current_hour = datetime.utcnow().hour

        # More likely to increase during peak hours
        if 14 <= current_hour < 16:  # Pre-peak
            trend_factor = random.uniform(0.6, 1.0)  # 60% chance of increase
        elif 16 <= current_hour < 21:  # Peak hours
            trend_factor = random.uniform(0.3,
                                          0.7)  # 30% chance of increase (already high)
        elif 21 <= current_hour < 23:  # Post-peak
            trend_factor = random.uniform(0.2, 0.6)  # 20% chance of increase
        else:  # Off-peak
            trend_factor = random.uniform(0.4, 0.8)  # 40% chance of increase

        if trend_factor > 0.6:
            return 'increasing'
        elif trend_factor < 0.4:
            return 'decreasing'
        else:
            return 'stable'

    async def get_billing_information(self) -> Dict[str, Any]:
        """Get billing structure information"""
        try:
            billing_info = {
                'utility_company': self.utility_company,
                'account_type': 'residential',
                'billing_cycle': 'monthly',
                'rate_schedule': 'TOU-D-PRIME',
                'pricing_structure': self.pricing_structure,
                'rate_tiers': self.price_tiers,
                'demand_charges': {
                    'winter': 8.50,
                    'summer': 15.75,
                    'spring_fall': 12.25
                },
                'fixed_charges': {
                    'customer_charge': 10.00,  # Monthly fixed charge
                    'meter_charge': 5.00,  # Monthly meter charge
                    'transmission_charge': 0.015,  # $/kWh
                    'distribution_charge': 0.025  # $/kWh
                },
                'peak_hours': {
                    'weekday': '16:00-21:00',
                    'weekend': 'none'
                },
                'off_peak_hours': {
                    'weekday': '22:00-06:00',
                    'weekend': 'all_day'
                },
                'taxes_and_fees': {
                    'state_tax': 0.08,  # 8% tax rate
                    'utility_tax': 0.02,  # 2% utility tax
                    'environmental_fee': 0.005  # $0.005/kWh
                },
                'net_metering': {
                    'available': True,
                    'credit_rate': 0.10,
                    # $0.10/kWh credit for excess generation
                    'monthly_rollover': True
                }
            }

            return billing_info

        except Exception as e:
            logger.error(f"Error getting billing information: {e}")
            return {}

    async def calculate_cost_estimate(self, consumption_kwh: float,
                                      time_period: str = 'current') -> Dict[
        str, Any]:
        """Calculate cost estimate for given consumption"""
        try:
            if time_period == 'current':
                pricing_data = await self.get_current_pricing()
            else:
                # For future periods, use forecast
                forecast = await self.get_pricing_forecast(24)
                pricing_data = forecast[
                    0] if forecast else await self.get_current_pricing()

            energy_charge = consumption_kwh * pricing_data.get('price_per_kwh',
                                                               self.base_rate)
            transmission_charge = consumption_kwh * pricing_data.get(
                'transmission_charge', 0.015)
            distribution_charge = consumption_kwh * pricing_data.get(
                'distribution_charge', 0.025)

            subtotal = energy_charge + transmission_charge + distribution_charge

            # Add taxes
            state_tax = subtotal * 0.08
            utility_tax = subtotal * 0.02
            environmental_fee = consumption_kwh * 0.005

            total_cost = subtotal + state_tax + utility_tax + environmental_fee

            cost_breakdown = {
                'consumption_kwh': consumption_kwh,
                'energy_charge': round(energy_charge, 4),
                'transmission_charge': round(transmission_charge, 4),
                'distribution_charge': round(distribution_charge, 4),
                'subtotal': round(subtotal, 4),
                'state_tax': round(state_tax, 4),
                'utility_tax': round(utility_tax, 4),
                'environmental_fee': round(environmental_fee, 4),
                'total_cost': round(total_cost, 4),
                'effective_rate': round(total_cost / consumption_kwh,
                                        4) if consumption_kwh > 0 else 0,
                'price_tier': pricing_data.get('price_tier', 'standard'),
                'calculation_time': datetime.utcnow().isoformat()
            }

            return cost_breakdown

        except Exception as e:
            logger.error(f"Error calculating cost estimate: {e}")
            return {}

    async def get_savings_opportunities(self) -> List[Dict[str, Any]]:
        """Get energy savings opportunities based on pricing"""
        try:
            opportunities = []
            current_time = datetime.utcnow()

            # Get pricing forecast
            forecast = await self.get_pricing_forecast(48)
            if not forecast:
                return opportunities

            # Find optimal time periods
            current_price = forecast[0]['price_per_kwh']

            # Identify low-price periods in next 24 hours
            low_price_periods = []
            for i, period in enumerate(forecast[:24]):
                if period[
                    'price_per_kwh'] < current_price * 0.8:  # 20% lower than current
                    low_price_periods.append({
                        'hour': (current_time + timedelta(hours=i)).hour,
                        'price': period['price_per_kwh'],
                        'savings_percent': ((current_price - period[
                            'price_per_kwh']) / current_price) * 100
                    })

            if low_price_periods:
                opportunities.append({
                    'type': 'time_shifting',
                    'title': 'Shift Energy Usage to Low-Price Periods',
                    'description': f'Energy prices will be lower during {len(low_price_periods)} periods in the next 24 hours',
                    'potential_savings': f"{max(p['savings_percent'] for p in low_price_periods):.1f}%",
                    'recommended_actions': [
                        'Schedule appliances for low-price periods',
                        'Pre-cool/heat home during cheaper hours',
                        'Charge electric vehicles during off-peak times'
                    ],
                    'optimal_periods': low_price_periods[:3]  # Top 3 periods
                })

            # Check for demand response opportunities
            dr_events = await self.get_demand_response_events()
            for event in dr_events:
                if not event.get('mandatory', False):  # Only voluntary events
                    opportunities.append({
                        'type': 'demand_response',
                        'title': f'Demand Response: {event["event_type"].replace("_", " ").title()}',
                        'description': event['description'],
                        'potential_savings': f"${event['incentive']:.3f}/kWh reduced",
                        'time_period': f"{event['start_time']} to {event['end_time']}",
                        'target_reduction': f"{event['target_reduction']}%",
                        'recommended_actions': [
                            'Reduce HVAC usage',
                            'Delay appliance operations',
                            'Use battery storage if available'
                        ]
                    })

            # Peak shaving opportunity
            peak_periods = [p for p in forecast[:24] if
                            p['price_tier'] == 'peak']
            if peak_periods:
                avg_peak_price = sum(
                    p['price_per_kwh'] for p in peak_periods) / len(
                    peak_periods)
                off_peak_periods = [p for p in forecast[:24] if
                                    p['price_tier'] == 'off_peak']

                if off_peak_periods:
                    avg_off_peak_price = sum(
                        p['price_per_kwh'] for p in off_peak_periods) / len(
                        off_peak_periods)
                    potential_savings = ((
                                                     avg_peak_price - avg_off_peak_price) / avg_peak_price) * 100

                    opportunities.append({
                        'type': 'peak_shaving',
                        'title': 'Peak Demand Shaving',
                        'description': f'Avoid usage during {len(peak_periods)} peak hours to save money',
                        'potential_savings': f"{potential_savings:.1f}%",
                        'peak_hours': [
                            datetime.fromisoformat(p['datetime']).hour for p in
                            peak_periods],
                        'recommended_actions': [
                            'Pre-condition home before peak hours',
                            'Use stored energy during peak periods',
                            'Minimize appliance usage 4-9 PM'
                        ]
                    })

            return opportunities

        except Exception as e:
            logger.error(f"Error getting savings opportunities: {e}")
            return []

    async def _update_pricing(self):
        """Update current pricing data"""
        try:
            current_time = datetime.utcnow()
            current_hour = current_time.hour

            # Determine price tier
            if 22 <= current_hour or current_hour < 6:
                tier = 'off_peak'
            elif 16 <= current_hour < 21:
                tier = 'peak'
            else:
                tier = 'standard'

            base_price = self.price_tiers[tier]

            # Add market variation
            market_factor = self._get_market_factor(current_time)
            price_variation = random.uniform(0.95,
                                             1.05)  # ±5% short-term variation

            current_price = base_price * market_factor * price_variation

            # Update pricing data
            self.current_pricing.update({
                'price_per_kwh': round(current_price, 4),
                'price_tier': tier,
                'timestamp': current_time.isoformat(),
                'demand_charge': self._calculate_demand_charge(),
                'market_factor': market_factor,
                'price_trend': self._calculate_price_trend()
            })

            self.last_update = current_time
            logger.debug(f"Updated pricing: ${current_price:.4f}/kWh ({tier})")

        except Exception as e:
            logger.error(f"Error updating pricing: {e}")

    async def health_check(self) -> bool:
        """Check if pricing service is healthy"""
        try:
            current_pricing = await self.get_current_pricing()
            return bool(current_pricing and 'price_per_kwh' in current_pricing)
        except Exception as e:
            logger.error(f"Pricing service health check failed: {e}")
            return False