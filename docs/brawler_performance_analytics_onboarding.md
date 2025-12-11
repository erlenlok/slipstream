# Brawler Performance Analytics Onboarding Guide

This guide explains how to set up, configure, and use the Brawler Performance Analytics system to track and monitor your market making strategy performance.

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation and Setup](#installation-and-setup)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [Dashboard Interface](#dashboard-interface)
8. [Alerting System](#alerting-system)
9. [Key Metrics Explained](#key-metrics-explained)
10. [Troubleshooting](#troubleshooting)

## Overview

The Brawler Performance Analytics system provides comprehensive monitoring and analysis of Brawler market making performance. It tracks key metrics including hit rates, markout analysis, PnL, inventory management, and risk metrics with both real-time and historical views.

## System Architecture

The system consists of several components:

- **Data Structures**: Core classes for trades, metrics, and analysis
- **Mock Data Pipeline**: Generates realistic trade data for testing
- **Core Metrics Calculator**: Calculates all primary performance metrics
- **Historical Analyzer**: Rolling window calculations and trend analysis
- **Per-Asset Analyzer**: Individual instrument performance tracking
- **Storage Layer**: TimescaleDB integration for data persistence
- **Dashboard**: Real-time web-based visualization
- **Alerting System**: Configurable threshold alerts and notifications

## Prerequisites

Before setting up the analytics system, ensure you have:

- Python 3.11 or higher
- PostgreSQL with TimescaleDB extension
- The Slipstream repository
- Dependencies specified in the analytics modules

## Installation and Setup

### 1. Environment Setup

First, activate your virtual environment:

```bash
cd slipstream
source .venv/bin/activate
```

### 2. Install Analytics Dependencies

If you haven't already, install the required packages:

```bash
pip install "fastapi[all]" uvicorn asyncpg aiohttp jinja2 --break-system-packages
```

### 3. Database Setup (Fast Start with Docker)

We provide a Docker-based setup for the analytics database (TimescaleDB). This creates a persistent database container isolated from your system.

**Prerequisites:**
- Docker Desktop or Docker Engine installed

**Setup:**

Run the setup script:

```bash
./scripts/setup_analytics_db.sh
```

This will:
1. Start a TimescaleDB container.
2. Mount a persistent volume `slipstream_db_data` (your data survives restarts).
3. Expose port `5432`.

**Manual Control:**

- Start: `docker-compose up -d`
- Stop: `docker-compose down`
- View Logs: `docker-compose logs -f`

**Data Persistence:**
All database data is stored in the Docker Volume named `slipstream_db_data`. Even if you remove the container, the volume persists. To delete data permanently, run `docker volume rm slipstream_db_data`.

## Configuration

### 1. Database Configuration

Create a database configuration for the analytics system:

```python
from slipstream.analytics.storage_layer import DatabaseConfig

db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="slipstream_analytics",
    username="postgres",
    password="your_password"
)
```

### 2. Alert Configuration

Set up the alert monitoring system:

```python
from slipstream.analytics.alerting_system import AlertMonitor, AlertThreshold, AlertSeverity, NotificationConfig

# Create notification configuration
notification_config = NotificationConfig(
    enabled_channels=["log"],  # Options: "log", "email", "webhook", "sms"
    email_config={  # Only if using email notifications
        "smtp_server": "smtp.gmail.com",
        "smtp_port": "587",
        "username": "your_email@gmail.com",
        "password": "your_password", 
        "recipient": "alerts@yourcompany.com"
    },
    webhook_urls=["https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"]
)

# Create alert monitor
alert_monitor = AlertMonitor(notification_config)

# Add thresholds for key metrics
alert_monitor.add_threshold(AlertThreshold(
    metric_name="hit_rate",
    threshold_value=60.0,  # Alert if hit rate drops below 60%
    operator="lt", 
    severity=AlertSeverity.HIGH
))

alert_monitor.add_threshold(AlertThreshold(
    metric_name="avg_markout_in",
    threshold_value=-0.001,  # Alert if maker markout becomes negative
    operator="lt",
    severity=AlertSeverity.MEDIUM
))

alert_monitor.add_threshold(AlertThreshold(
    metric_name="max_drawdown",
    threshold_value=0.05,  # Alert if drawdown exceeds 5%
    operator="gt",
    severity=AlertSeverity.CRITICAL
))
```

## Running the System

### 1. Initialize the Analytics Service

```python
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.historical_analyzer import HistoricalAnalyzer
from slipstream.analytics.per_asset_analyzer import PerAssetPerformanceAnalyzer
from slipstream.analytics.storage_layer import AnalyticsStorage
from slipstream.analytics.dashboard import RealTimeDashboard

# Initialize components
core_calculator = CoreMetricsCalculator()
historical_analyzer = HistoricalAnalyzer()
per_asset_analyzer = PerAssetPerformanceAnalyzer()
storage = AnalyticsStorage(db_config)

# Connect to storage
await storage.connect()
await storage.create_tables()
```

### 2. Processing Trades

To track performance, you need to feed trade events to the analytics system:

```python
from slipstream.analytics.data_structures import TradeEvent, TradeType

# Example trade event
trade = TradeEvent(
    timestamp=datetime.now(),
    symbol="BTC",
    side="buy",
    quantity=0.1,
    price=50000.0,
    trade_type=TradeType.MAKER,  # or TradeType.TAKER
    reference_price=50010.0,  # For markout calculation
    fees_paid=5.0,
    funding_paid=2.0,
    position_before=0.0,
    position_after=0.1
)

# Process the trade through all systems
core_calculator.process_trade(trade)
per_asset_analyzer.per_asset.add_trade(trade)

# Check for alerts
alerts = await alert_monitor.check_metrics(core_calculator.calculate_final_metrics(), asset=trade.symbol)

# Store the trade event
await storage.store_trade_event(trade)
```

### 3. Running the Dashboard

To start the real-time dashboard:

```python
from slipstream.analytics.dashboard import DashboardConfig

# Configure dashboard
dashboard_config = DashboardConfig(
    host="0.0.0.0",
    port=8000,
    log_level="info"
)

# Create and start dashboard
dashboard = RealTimeDashboard(config=dashboard_config)

# Start the server
dashboard.start_server()
```

The dashboard will be available at `http://localhost:8000` (or your configured host/port).

## Dashboard Interface

### Main Dashboard View
- **Current Metrics**: Real-time display of hit rate, PnL, markout, Sharpe ratio, etc.
- **Charts**: Performance over time, hit rate trends, inventory levels
- **Per-Asset Breakdown**: Performance metrics for each instrument
- **Alert Panel**: Recent alerts and notifications

### Navigation
- **Metrics Overview**: High-level performance indicators
- **Historical Trends**: Charts showing metrics over time
- **Per-Asset Analysis**: Individual instrument performance
- **Alerts Log**: All triggered alerts and notifications

## Alerting System

### Configuring Alerts

Different types of alerts you can configure:

- **Hit Rate Degradation**: When hit rates fall below thresholds
- **Markout Negative Trends**: When maker/taker markout turns negative
- **PnL Thresholds**: When PnL crosses critical levels
- **Inventory Concentration**: When inventory becomes too concentrated in one asset
- **Max Drawdown**: When drawdown exceeds acceptable levels
- **Sharpe Ratio Drops**: When risk-adjusted returns deteriorate

### Alert Severity Levels

- **LOW**: Minor performance degradations, informational
- **MEDIUM**: Notable performance changes requiring attention
- **HIGH**: Significant performance issues requiring immediate attention
- **CRITICAL**: Critical problems requiring emergency response

### Notification Channels

Configure how you want to receive alerts:

- **Log**: Alerts logged to application logs
- **Email**: Alerts sent via email
- **Webhook**: Alerts sent to webhook URLs (like Slack)
- **SMS**: Alerts sent via SMS (requires provider configuration)

## Key Metrics Explained

### Hit Rate Metrics
- **Definition**: Percentage of quotes that result in fills
- **Importance**: Measures liquidity provision effectiveness
- **Maker vs Taker**: Differentiates between passive (maker) and aggressive (taker) fills

### Markout Analysis
- **Maker Markout**: Profit/loss when providing liquidity (passive fills)
- **Taker Markout**: Profit/loss when taking liquidity (aggressive fills)  
- **Calculation**: Difference between fill price and reference price, net of fees

### PnL Metrics
- **Gross PnL**: Revenue from trading before costs
- **Net PnL**: Gross PnL minus fees and funding costs
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown**: Peak-to-trough decline in PnL
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Potential loss at given confidence level

### Inventory Metrics
- **Average Inventory**: Average absolute position held
- **Max Inventory**: Peak inventory exposure
- **Inventory Turnover**: How frequently positions are cycled
- **Concentration**: How inventory is distributed across instruments

## Troubleshooting

### Common Issues

#### Database Connection Issues
- Ensure PostgreSQL is running
- Verify database credentials in configuration
- Check that TimescaleDB extension is installed
- Confirm tables were created successfully

#### Dashboard Not Starting
- Check if required ports are available
- Verify FastAPI and related dependencies are installed
- Look for Python import errors in the logs

#### Alerts Not Triggering
- Verify threshold values are properly set
- Check that alerts are enabled
- Ensure notification channels are configured correctly

#### Performance Issues
- Check database query performance
- Monitor system resource usage
- Consider indexing strategies for historical data

### Support Configuration

For additional support or to extend the analytics system, you can:

1. Review the source code in `src/slipstream/analytics/`
2. Check the configuration files for options
3. Monitor logs for detailed error information
4. Consult the integration tests for usage examples

## Next Steps

1. Configure your specific alert thresholds based on your strategy parameters
2. Set up proper notification channels (email, webhook, etc.)
3. Integrate the analytics system with your Brawler execution system
4. Set up data retention policies for historical data
5. Schedule regular reviews of performance metrics
6. Create custom dashboards for your specific needs

## Additional Resources

- Source code: `src/slipstream/analytics/`
- Test examples: `tests/test_*.py` files
- Configuration examples are provided in the code
- Integration examples in `src/slipstream/analytics/integration_tests.py`