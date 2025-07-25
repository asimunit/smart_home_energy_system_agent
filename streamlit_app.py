#!/usr/bin/env python3
"""
Smart Home Energy Management System - Streamlit Dashboard
Interactive web interface for system monitoring and control
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import asyncio
import websocket
import threading
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

# Page configuration
st.set_page_config(
    page_title="Smart Home Energy Management",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    .device-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []


class APIClient:
    """Client for interacting with the FastAPI backend"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request to API"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API GET error for {endpoint}: {e}")
            return {}

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to API"""
        try:
            response = requests.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API POST error for {endpoint}: {e}")
            return {}

    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make PUT request to API"""
        try:
            response = requests.put(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API PUT error for {endpoint}: {e}")
            return {}


# Initialize API client
api_client = APIClient(API_BASE_URL)


def get_system_status():
    """Get system status from API"""
    return api_client.get("/system/status")


def get_agents():
    """Get agent information from API"""
    return api_client.get("/system/agents")


def get_devices():
    """Get device information from API"""
    return api_client.get("/devices")


def get_current_energy():
    """Get current energy data from API"""
    return api_client.get("/energy/current")


def get_energy_data(hours_back: int = 24):
    """Get historical energy data"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_back)

    query_data = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "aggregation": "hourly"
    }

    return api_client.post("/energy/query", query_data)


def control_device(device_id: str, action: str,
                   parameters: Dict[str, Any] = None):
    """Control a device"""
    if parameters is None:
        parameters = {}

    control_data = {
        "device_id": device_id,
        "action": action,
        "parameters": parameters
    }

    return api_client.post("/devices/control", control_data)


def get_alerts():
    """Get system alerts"""
    return api_client.get("/alerts")


def resolve_alert(alert_id: str):
    """Resolve an alert"""
    return api_client.post(f"/alerts/{alert_id}/resolve", {})


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🏠 Smart Home Energy Management System")
    st.markdown("Real-time monitoring and control dashboard")
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Energy Analytics", "Device Control",
         "System Administration", "User Preferences", "Alerts"]
    )

    # Connection status
    with st.sidebar:
        st.subheader("System Status")
        try:
            system_status = get_system_status()
            if system_status.get('status') == 'running':
                st.success("✅ System Online")
            else:
                st.error("❌ System Offline")
        except:
            st.error("❌ Cannot connect to system")

    # Route to different pages
    if page == "Dashboard":
        dashboard_page()
    elif page == "Energy Analytics":
        energy_analytics_page()
    elif page == "Device Control":
        device_control_page()
    elif page == "System Administration":
        system_admin_page()
    elif page == "User Preferences":
        user_preferences_page()
    elif page == "Alerts":
        alerts_page()


def dashboard_page():
    """Main dashboard page"""
    st.header("📊 System Dashboard")

    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)

    try:
        current_energy = get_current_energy()

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Current Consumption",
                value=f"{current_energy.get('total_consumption', 0):.1f} kW",
                delta=f"{current_energy.get('net_consumption', 0):.1f} kW net"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Solar Generation",
                value=f"{current_energy.get('total_generation', 0):.1f} kW",
                delta="↗️ Generating"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Battery Charge",
                value=f"{current_energy.get('battery_charge', 0):.0f}%",
                delta="🔋 Charging" if current_energy.get('net_consumption',
                                                         0) < 0 else "🔋 Discharging"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Current Rate",
                value=f"${current_energy.get('current_cost_rate', 0):.3f}/kWh",
                delta="📈 Peak" if current_energy.get('current_cost_rate',
                                                     0) > 0.15 else "📉 Off-Peak"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading energy data: {e}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy Flow (Last 24 Hours)")
        try:
            energy_data = get_energy_data(24)
            if energy_data.get('data'):
                df = pd.DataFrame(energy_data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['consumption'],
                    mode='lines',
                    name='Consumption',
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['generation'],
                    mode='lines',
                    name='Generation',
                    line=dict(color='green')
                ))

                fig.update_layout(
                    title="Energy Consumption vs Generation",
                    xaxis_title="Time",
                    yaxis_title="Power (kW)",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No energy data available")
        except Exception as e:
            st.error(f"Error loading energy chart: {e}")

    with col2:
        st.subheader("System Agents Status")
        try:
            agents = get_agents()
            if agents:
                agent_df = pd.DataFrame(agents)

                # Pie chart of agent status
                status_counts = agent_df['status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Agent Status Distribution",
                    color_discrete_map={'running': 'green', 'stopped': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Agent list
                st.subheader("Agent Details")
                for agent in agents:
                    status_color = "🟢" if agent['is_running'] else "🔴"
                    st.write(
                        f"{status_color} **{agent['agent_type'].title()}**: {agent['status']}")
            else:
                st.info("No agent data available")
        except Exception as e:
            st.error(f"Error loading agent data: {e}")

    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    try:
        alerts = get_alerts()
        if alerts:
            recent_alerts = sorted(alerts, key=lambda x: x['timestamp'],
                                   reverse=True)[:5]
            for alert in recent_alerts:
                alert_class = f"alert-{alert['severity']}"
                st.markdown(f'''
                <div class="alert-card {alert_class}">
                    <strong>{alert['component'].title()}</strong> - {alert['message']}
                    <br><small>{alert['timestamp']}</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.success("No active alerts")
    except Exception as e:
        st.error(f"Error loading alerts: {e}")


def energy_analytics_page():
    """Energy analytics page"""
    st.header("⚡ Energy Analytics")

    # Time range selector
    col1, col2, col3 = st.columns(3)
    with col1:
        time_range = st.selectbox("Time Range",
                                  ["24 Hours", "7 Days", "30 Days"])
    with col2:
        aggregation = st.selectbox("Aggregation",
                                   ["Hourly", "Daily", "Weekly"])
    with col3:
        if st.button("Refresh Data"):
            st.rerun()

    # Convert time range to hours
    hours_map = {"24 Hours": 24, "7 Days": 168, "30 Days": 720}
    hours_back = hours_map.get(time_range, 24)

    try:
        energy_data = get_energy_data(hours_back)
        if energy_data.get('data'):
            df = pd.DataFrame(energy_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Main energy chart
            st.subheader("Energy Consumption and Generation")
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Power (kW)', 'Cost ($)'),
                vertical_spacing=0.1
            )

            # Power chart
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['consumption'],
                           name='Consumption', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['generation'],
                           name='Generation', line=dict(color='green')),
                row=1, col=1
            )

            # Cost chart
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cost'], name='Cost',
                           line=dict(color='blue')),
                row=2, col=1
            )

            fig.update_layout(height=600,
                              title_text="Energy Analytics Overview")
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_consumption = df['consumption'].sum()
                st.metric("Total Consumption", f"{total_consumption:.1f} kWh")

            with col2:
                total_generation = df['generation'].sum()
                st.metric("Total Generation", f"{total_generation:.1f} kWh")

            with col3:
                total_cost = df['cost'].sum()
                st.metric("Total Cost", f"${total_cost:.2f}")

            with col4:
                net_consumption = total_consumption - total_generation
                st.metric("Net Consumption", f"{net_consumption:.1f} kWh")

            # Efficiency metrics
            st.subheader("Efficiency Metrics")
            col1, col2 = st.columns(2)

            with col1:
                if total_consumption > 0:
                    self_consumption_ratio = min(1.0,
                                                 total_generation / total_consumption)
                    st.metric("Self-Consumption Ratio",
                              f"{self_consumption_ratio:.1%}")

                avg_consumption = df['consumption'].mean()
                st.metric("Average Consumption", f"{avg_consumption:.2f} kW")

            with col2:
                if total_consumption > 0:
                    avg_cost_per_kwh = total_cost / total_consumption
                    st.metric("Average Cost per kWh",
                              f"${avg_cost_per_kwh:.3f}")

                peak_consumption = df['consumption'].max()
                st.metric("Peak Consumption", f"{peak_consumption:.2f} kW")

        else:
            st.info("No energy data available for the selected time range")

    except Exception as e:
        st.error(f"Error loading energy analytics: {e}")


def device_control_page():
    """Device control page"""
    st.header("🏠 Device Control")

    try:
        devices_data = get_devices()
        devices = devices_data.get('devices', [])

        if devices:
            # Device grid
            cols = st.columns(2)
            for i, device in enumerate(devices):
                with cols[i % 2]:
                    st.markdown('<div class="device-card">',
                                unsafe_allow_html=True)

                    # Device header
                    status_icon = "🟢" if device['status'] == 'online' else "🔴"
                    st.subheader(f"{status_icon} {device['name']}")
                    st.write(f"**Type:** {device['device_type'].title()}")
                    st.write(f"**Power:** {device['power_consumption']} W")

                    # Device-specific controls
                    if device['device_type'] == 'hvac':
                        col1, col2 = st.columns(2)
                        with col1:
                            temp_setting = st.slider(
                                "Temperature",
                                min_value=60,
                                max_value=85,
                                value=72,
                                key=f"temp_{device['device_id']}"
                            )
                        with col2:
                            fan_speed = st.selectbox(
                                "Fan Speed",
                                ["Auto", "Low", "Medium", "High"],
                                key=f"fan_{device['device_id']}"
                            )

                        if st.button(f"Update HVAC",
                                     key=f"update_hvac_{device['device_id']}"):
                            result = control_device(
                                device['device_id'],
                                "set_temperature",
                                {"temperature": temp_setting,
                                 "fan_speed": fan_speed.lower()}
                            )
                            if result:
                                st.success("HVAC settings updated!")

                    elif device['device_type'] == 'lighting':
                        col1, col2 = st.columns(2)
                        with col1:
                            brightness = st.slider(
                                "Brightness",
                                min_value=0,
                                max_value=100,
                                value=75,
                                key=f"brightness_{device['device_id']}"
                            )
                        with col2:
                            color_temp = st.slider(
                                "Color Temperature (K)",
                                min_value=2200,
                                max_value=6500,
                                value=4000,
                                key=f"color_{device['device_id']}"
                            )

                        if st.button(f"Update Lighting",
                                     key=f"update_light_{device['device_id']}"):
                            result = control_device(
                                device['device_id'],
                                "set_lighting",
                                {"brightness": brightness,
                                 "color_temperature": color_temp}
                            )
                            if result:
                                st.success("Lighting settings updated!")

                    # Common controls
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Turn On",
                                     key=f"on_{device['device_id']}"):
                            result = control_device(device['device_id'],
                                                    "turn_on")
                            if result:
                                st.success("Device turned on!")

                    with col2:
                        if st.button(f"Turn Off",
                                     key=f"off_{device['device_id']}"):
                            result = control_device(device['device_id'],
                                                    "turn_off")
                            if result:
                                st.success("Device turned off!")

                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No devices found")

    except Exception as e:
        st.error(f"Error loading devices: {e}")


def system_admin_page():
    """System administration page"""
    st.header("⚙️ System Administration")

    # System status
    st.subheader("System Status")
    try:
        system_status = get_system_status()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System Status",
                      system_status.get('status', 'unknown').title())
        with col2:
            uptime = system_status.get('uptime_seconds', 0)
            uptime_hours = uptime / 3600
            st.metric("Uptime", f"{uptime_hours:.1f} hours")
        with col3:
            healthy_agents = system_status.get('healthy_agents', 0)
            total_agents = system_status.get('agents_count', 0)
            st.metric("Agent Health", f"{healthy_agents}/{total_agents}")

        # System controls
        st.subheader("System Controls")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Restart System", type="secondary"):
                result = api_client.post("/system/restart", {})
                if result:
                    st.success("System restart initiated")

        with col2:
            if st.button("Refresh Status"):
                st.rerun()

        with col3:
            if st.button("Export Logs", type="secondary"):
                st.info("Log export functionality would be implemented here")

    except Exception as e:
        st.error(f"Error loading system status: {e}")

    # Agent management
    st.subheader("Agent Management")
    try:
        agents = get_agents()
        if agents:
            agent_df = pd.DataFrame(agents)

            # Display agents table
            st.dataframe(
                agent_df[['agent_type', 'status', 'is_running']],
                use_container_width=True
            )

            # Agent controls
            selected_agent = st.selectbox("Select Agent",
                                          [agent['agent_id'] for agent in
                                           agents])

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Start Agent"):
                    st.info(
                        "Agent start functionality would be implemented here")
            with col2:
                if st.button("Stop Agent"):
                    st.info(
                        "Agent stop functionality would be implemented here")
            with col3:
                if st.button("Restart Agent"):
                    st.info(
                        "Agent restart functionality would be implemented here")

        else:
            st.info("No agent data available")

    except Exception as e:
        st.error(f"Error loading agent data: {e}")

    # Performance metrics
    st.subheader("Performance Metrics")
    try:
        # This would show real-time performance data
        # For now, show placeholder metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("CPU Usage", "45%", "↗️ +5%")
        with col2:
            st.metric("Memory Usage", "62%", "↘️ -3%")
        with col3:
            st.metric("Disk Usage", "78%", "↗️ +1%")

    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")


def user_preferences_page():
    """User preferences page"""
    st.header("👤 User Preferences")

    # User selection
    user_id = st.selectbox("Select User", ["user_001", "user_002", "admin"])

    try:
        preferences = api_client.get(f"/users/{user_id}/preferences")

        if preferences:
            # Temperature preferences
            st.subheader("🌡️ Temperature Preferences")
            col1, col2 = st.columns(2)

            with col1:
                heating_setpoint = st.slider(
                    "Heating Setpoint (°F)",
                    min_value=60,
                    max_value=80,
                    value=int(preferences.get('temperature', {}).get(
                        'heating_setpoint', 70))
                )

            with col2:
                cooling_setpoint = st.slider(
                    "Cooling Setpoint (°F)",
                    min_value=70,
                    max_value=85,
                    value=int(preferences.get('temperature', {}).get(
                        'cooling_setpoint', 76))
                )

            # Energy preferences
            st.subheader("⚡ Energy Preferences")
            col1, col2 = st.columns(2)

            with col1:
                priority = st.selectbox(
                    "Energy Priority",
                    ["comfort_first", "balanced", "efficiency_first"],
                    index=1
                )

            with col2:
                budget_target = st.number_input(
                    "Monthly Budget Target ($)",
                    min_value=0.0,
                    value=float(
                        preferences.get('energy', {}).get('budget_target',
                                                          150))
                )

            # Lighting preferences
            st.subheader("💡 Lighting Preferences")
            col1, col2 = st.columns(2)

            with col1:
                brightness = st.slider(
                    "Default Brightness (%)",
                    min_value=0,
                    max_value=100,
                    value=preferences.get('lighting', {}).get(
                        'brightness_preference', 75)
                )

            with col2:
                circadian_lighting = st.checkbox(
                    "Enable Circadian Lighting",
                    value=preferences.get('lighting', {}).get(
                        'circadian_lighting', True)
                )

            # Notification preferences
            st.subheader("🔔 Notification Preferences")
            col1, col2 = st.columns(2)

            with col1:
                energy_alerts = st.checkbox(
                    "Energy Alerts",
                    value=preferences.get('notifications', {}).get(
                        'energy_alerts_enabled', True)
                )
                maintenance_reminders = st.checkbox(
                    "Maintenance Reminders",
                    value=preferences.get('notifications', {}).get(
                        'maintenance_reminders', True)
                )

            with col2:
                efficiency_tips = st.checkbox(
                    "Efficiency Tips",
                    value=preferences.get('notifications', {}).get(
                        'efficiency_tips', True)
                )
                cost_alerts = st.checkbox(
                    "Cost Alerts",
                    value=preferences.get('notifications', {}).get(
                        'cost_alerts', True)
                )

            # Save preferences
            if st.button("Save Preferences", type="primary"):
                updated_preferences = {
                    "temperature": {
                        "heating_setpoint": heating_setpoint,
                        "cooling_setpoint": cooling_setpoint
                    },
                    "energy": {
                        "priority": priority,
                        "budget_target": budget_target
                    },
                    "lighting": {
                        "brightness_preference": brightness,
                        "circadian_lighting": circadian_lighting
                    },
                    "notifications": {
                        "energy_alerts_enabled": energy_alerts,
                        "maintenance_reminders": maintenance_reminders,
                        "efficiency_tips": efficiency_tips,
                        "cost_alerts": cost_alerts
                    }
                }

                result = api_client.put(
                    f"/users/{user_id}/preferences",
                    {
                        "user_id": user_id,
                        "category": "all",
                        "preferences": updated_preferences
                    }
                )

                if result:
                    st.success("Preferences saved successfully!")
                else:
                    st.error("Failed to save preferences")

        else:
            st.info("No preferences found for this user")

    except Exception as e:
        st.error(f"Error loading user preferences: {e}")


def alerts_page():
    """Alerts management page"""
    st.header("🚨 System Alerts")

    try:
        alerts = get_alerts()

        if alerts:
            # Alert filters
            col1, col2, col3 = st.columns(3)

            with col1:
                severity_filter = st.selectbox(
                    "Filter by Severity",
                    ["All", "critical", "warning", "info"]
                )

            with col2:
                status_filter = st.selectbox(
                    "Filter by Status",
                    ["All", "Active", "Resolved"]
                )

            with col3:
                if st.button("Refresh Alerts"):
                    st.rerun()

            # Filter alerts
            filtered_alerts = alerts
            if severity_filter != "All":
                filtered_alerts = [a for a in filtered_alerts if
                                   a['severity'] == severity_filter]
            if status_filter == "Active":
                filtered_alerts = [a for a in filtered_alerts if
                                   not a['resolved']]
            elif status_filter == "Resolved":
                filtered_alerts = [a for a in filtered_alerts if a['resolved']]

            # Display alerts
            for alert in sorted(filtered_alerts, key=lambda x: x['timestamp'],
                                reverse=True):
                severity_color = {
                    'critical': '🔴',
                    'warning': '🟠',
                    'info': '🔵'
                }.get(alert['severity'], '🔵')

                status_text = "✅ Resolved" if alert[
                    'resolved'] else "⚠️ Active"

                with st.expander(
                        f"{severity_color} {alert['message']} - {status_text}"):
                    st.write(f"**Component:** {alert['component']}")
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Timestamp:** {alert['timestamp']}")
                    st.write(f"**Alert ID:** {alert['alert_id']}")

                    if not alert['resolved']:
                        if st.button(f"Resolve Alert",
                                     key=f"resolve_{alert['alert_id']}"):
                            result = resolve_alert(alert['alert_id'])
                            if result:
                                st.success("Alert resolved!")
                                st.rerun()

        else:
            st.success("🎉 No alerts - system is running smoothly!")

    except Exception as e:
        st.error(f"Error loading alerts: {e}")


if __name__ == "__main__":
    main()