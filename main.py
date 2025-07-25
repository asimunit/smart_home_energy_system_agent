#!/usr/bin/env python3
"""
Smart Home Energy Management System
Main application entry point
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any

# Import configuration and utilities
from config.settings import settings
from config.elasticsearch_config import es_config
from utils.logger import setup_logger, LogExecutionTime, audit_logger

# Import all agents
from agents.energy_monitor_agent import EnergyMonitorAgent
from agents.price_intelligence_agent import PriceIntelligenceAgent
from agents.hvac_agent import HVACAgent
from agents.appliance_agent import ApplianceAgent
from agents.lighting_agent import LightingAgent
from agents.ev_charging_agent import EVChargingAgent
from agents.solar_battery_agent import SolarBatteryAgent
from agents.comfort_optimization_agent import ComfortOptimizationAgent
from agents.negotiator_agent import NegotiatorAgent

# Import services
from services.device_service import DeviceService
from services.weather_service import WeatherService
from services.pricing_service import PricingService

logger = setup_logger("main")


class SmartHomeEnergySystem:
    """Main system orchestrator"""

    def __init__(self):
        self.agents: List[Any] = []
        self.services: Dict[str, Any] = {}
        self.running = False
        self.startup_time = None

        # Initialize services
        self.services['device'] = DeviceService()
        self.services['weather'] = WeatherService()
        self.services['pricing'] = PricingService()

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all system agents"""
        try:
            logger.info("Initializing system agents...")

            # Create agent instances
            self.agents = [
                EnergyMonitorAgent(),
                PriceIntelligenceAgent(),
                HVACAgent(),
                ApplianceAgent(),
                LightingAgent(),
                EVChargingAgent(),
                SolarBatteryAgent(),
                ComfortOptimizationAgent(),
                NegotiatorAgent()  # Negotiator should be last
            ]

            logger.info(f"Created {len(self.agents)} agents")


        except Exception as e:

            logger.error(f"Failed to initialize agents: {e}", exc_info=True)

            raise

    async def startup(self):
        """Start the smart home energy system"""
        try:
            self.startup_time = datetime.utcnow()
            logger.info("Starting Smart Home Energy Management System...")

            # Check system health
            await self._health_check()

            # Initialize database indices
            await self._setup_database()

            # Initialize all agents
            await self._start_agents()

            # Start system monitoring
            await self._start_monitoring()

            self.running = True

            # Log system startup
            audit_logger.log_system_change(
                component="system",
                change_type="startup",
                new_value="running"
            )

            startup_duration = (
                        datetime.utcnow() - self.startup_time).total_seconds()
            logger.info(
                f"Smart Home Energy System started successfully in {startup_duration:.2f} seconds")

            # Print system status
            await self._print_system_status()

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.shutdown()
            raise

    async def _health_check(self):
        """Perform system health checks"""
        logger.info("Performing system health checks...")

        # Check Elasticsearch
        if not es_config.health_check():
            raise RuntimeError("Elasticsearch is not healthy")

        # Check services
        for service_name, service in self.services.items():
            if hasattr(service, 'health_check'):
                if not await service.health_check():
                    logger.warning(
                        f"Service {service_name} health check failed")

        logger.info("System health checks passed")

    async def _setup_database(self):
        """Setup database indices and schemas"""
        logger.info("Setting up database...")

        try:
            # Create Elasticsearch indices
            es_config.create_indices()
            logger.info("Database indices created successfully")

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    async def _start_agents(self):
        """Start all system agents"""
        logger.info("Starting system agents...")

        successful_agents = 0
        failed_agents = 0

        for agent in self.agents:
            try:
                with LogExecutionTime(logger,
                                      f"Starting agent {agent.agent_id}"):
                    await agent.start()
                    successful_agents += 1
                    logger.info(
                        f"✓ Agent {agent.agent_id} started successfully")

            except Exception as e:
                failed_agents += 1
                logger.error(f"✗ Failed to start agent {agent.agent_id}: {e}")

        logger.info(
            f"Agent startup complete: {successful_agents} successful, {failed_agents} failed")

        if failed_agents > 0:
            logger.warning(
                f"{failed_agents} agents failed to start - system may have reduced functionality")

    async def _start_monitoring(self):
        """Start system monitoring tasks"""
        logger.info("Starting system monitoring...")

        # Create monitoring tasks
        asyncio.create_task(self._system_health_monitor())
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._resource_monitor())

        logger.info("System monitoring started")

    async def _system_health_monitor(self):
        """Monitor overall system health"""
        while self.running:
            try:
                # Check agent health
                healthy_agents = 0
                total_agents = len(self.agents)

                for agent in self.agents:
                    if agent.is_running:
                        healthy_agents += 1

                health_percentage = (
                                                healthy_agents / total_agents) * 100 if total_agents > 0 else 0

                if health_percentage < 80:
                    logger.warning(
                        f"System health degraded: {health_percentage:.1f}% agents healthy")

                # Check service health
                for service_name, service in self.services.items():
                    if hasattr(service, 'health_check'):
                        try:
                            if not await service.health_check():
                                logger.warning(
                                    f"Service {service_name} health check failed")
                        except Exception as e:
                            logger.error(
                                f"Health check error for {service_name}: {e}")

                # Sleep for 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Short sleep on error

    async def _performance_monitor(self):
        """Monitor system performance"""
        while self.running:
            try:
                # Monitor memory usage, CPU usage, etc.
                # This is a simplified version - in production you'd use more sophisticated monitoring

                import psutil

                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                if cpu_percent > 80:
                    logger.warning(f"High CPU usage: {cpu_percent}%")

                if memory.percent > 80:
                    logger.warning(f"High memory usage: {memory.percent}%")

                if disk.percent > 90:
                    logger.warning(f"High disk usage: {disk.percent}%")

                # Log performance metrics every 10 minutes
                await asyncio.sleep(600)

            except ImportError:
                # psutil not available, skip performance monitoring
                logger.info(
                    "psutil not available, skipping performance monitoring")
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)

    async def _resource_monitor(self):
        """Monitor system resources and limits"""
        while self.running:
            try:
                # Monitor database connections, memory usage, etc.
                # This is a placeholder for more sophisticated resource monitoring

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(60)

    async def _print_system_status(self):
        """Print current system status"""
        print("\n" + "=" * 80)
        print("SMART HOME ENERGY MANAGEMENT SYSTEM")
        print("=" * 80)
        print(f"Status: RUNNING")
        print(
            f"Started: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Agents: {len(self.agents)} initialized")
        print(f"Services: {len(self.services)} loaded")
        print()

        print("ACTIVE AGENTS:")
        print("-" * 40)
        for agent in self.agents:
            status = "RUNNING" if agent.is_running else "STOPPED"
            print(f"• {agent.agent_type:<20} [{status}]")

        print()
        print("AVAILABLE SERVICES:")
        print("-" * 40)
        for service_name in self.services:
            print(f"• {service_name.title()} Service")

        print()
        print("SYSTEM CONFIGURATION:")
        print("-" * 40)
        print(
            f"• Elasticsearch: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
        print(f"• Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        print(f"• LLM Model: {settings.GEMINI_MODEL}")
        print(f"• Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"• Log Level: {settings.LOG_LEVEL}")
        print(f"• Debug Mode: {settings.DEBUG}")

        print("\n" + "=" * 80)
        print("System ready for operation")
        print("Press Ctrl+C to shutdown gracefully")
        print("=" * 80 + "\n")

    async def run(self):
        """Main run loop"""
        try:
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"System error in main loop: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the system"""
        if not self.running:
            return

        logger.info("Shutting down Smart Home Energy System...")
        self.running = False

        # Stop all agents
        stopped_agents = 0
        for agent in self.agents:
            try:
                await agent.stop()
                stopped_agents += 1
                logger.info(f"✓ Agent {agent.agent_id} stopped")
            except Exception as e:
                logger.error(f"✗ Error stopping agent {agent.agent_id}: {e}")

        # Cleanup services
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"✓ Service {service_name} cleaned up")
            except Exception as e:
                logger.error(
                    f"✗ Error cleaning up service {service_name}: {e}")

        # Log system shutdown
        audit_logger.log_system_change(
            component="system",
            change_type="shutdown",
            old_value="running",
            new_value="stopped"
        )

        shutdown_time = datetime.utcnow()
        if self.startup_time:
            uptime = (shutdown_time - self.startup_time).total_seconds()
            logger.info(f"System uptime: {uptime:.1f} seconds")

        logger.info(
            f"Smart Home Energy System shutdown complete ({stopped_agents} agents stopped)")


def setup_signal_handlers(system: SmartHomeEnergySystem):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(system.shutdown())

    # Handle common shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Handle other signals if available (Unix only)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)


async def main():
    """Main application entry point"""
    try:
        # Create system instance
        system = SmartHomeEnergySystem()

        # Setup signal handlers
        setup_signal_handlers(system)

        # Start system
        await system.startup()

        # Run main loop
        await system.run()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal application error: {e}")
        sys.exit(1)


def cli_main():
    """CLI entry point"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("Error: Python 3.8 or higher is required")
            sys.exit(1)

        # Run the application
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()