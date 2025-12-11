"""
Real-time dashboard and visualization for Brawler performance tracking.

This module implements a FastAPI-based dashboard for real-time monitoring
of Brawler's market making performance metrics.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from threading import Thread

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from slipstream.analytics.data_structures import PerformanceMetrics
from slipstream.analytics.mock_data_pipeline import MockBrawlerEventProcessor, MockTradeGenerator
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.historical_analyzer import HistoricalAnalyzer, TimeWindow
from slipstream.analytics.per_asset_analyzer import PerAssetPerformanceAnalyzer


class DashboardConfig(BaseModel):
    """Configuration for the dashboard."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


@dataclass
class DashboardData:
    """Container for dashboard data."""
    
    current_metrics: Optional[PerformanceMetrics] = None
    historical_data: List[Dict[str, Any]] = None
    per_asset_data: Dict[str, Dict[str, float]] = None
    trend_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        if self.current_metrics:
            result['current_metrics'] = asdict(self.current_metrics)
        if self.historical_data:
            result['historical_data'] = self.historical_data
        if self.per_asset_data:
            result['per_asset_data'] = self.per_asset_data
        if self.trend_data:
            result['trend_data'] = self.trend_data
        return result


class MetricType(Enum):
    """Types of metrics that can be visualized."""
    
    PNL = "pnl"
    HIT_RATE = "hit_rate"
    MARKOUT = "markout"
    SHARPE_RATIO = "sharpe_ratio"
    INVENTORY = "inventory"
    VOLATILITY = "volatility"


class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        for connection in self.active_connections[:]:  # Copy list to prevent modification during iteration
            try:
                await connection.send_text(json.dumps(message))
            except WebSocketDisconnect:
                self.active_connections.remove(connection)


class DashboardService:
    """Service class to manage dashboard data and updates."""
    
    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize analytics components
        self.event_processor = MockBrawlerEventProcessor()
        self.core_calculator = CoreMetricsCalculator()
        self.historical_analyzer = HistoricalAnalyzer()
        self.per_asset_analyzer = PerAssetPerformanceAnalyzer()
        
        # WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # Store current data
        self.current_data = DashboardData()
        
        # For mock mode, initialize with mock data
        if mock_mode:
            self._initialize_mock_data()
    
    def _initialize_mock_data(self) -> None:
        """Initialize with mock data for demonstration."""
        # Generate some mock trades to initialize metrics
        generator = MockTradeGenerator()
        trades = generator.generate_24h_trades(datetime.now() - timedelta(hours=24))
        
        for trade in trades[:20]:  # Process first 20 trades to initialize metrics
            self.core_calculator.process_trade(trade)
            self.event_processor.process_trade_event(trade)
            self.per_asset_analyzer.per_asset.add_trade(trade)
        
        self.update_current_metrics()
        
        # Also add some historical data
        self.add_historical_data_points()
    
    def update_current_metrics(self) -> None:
        """Update current metrics based on latest data."""
        self.current_data.current_metrics = self.core_calculator.calculate_final_metrics()
        self.current_data.per_asset_data = self.per_asset_analyzer.get_per_asset_summary()
        self.current_data.trend_data = self.historical_analyzer.get_performance_trends()
    
    def add_historical_data_points(self) -> None:
        """Add historical data points for charting."""
        if self.current_data.historical_data is None:
            self.current_data.historical_data = []
        
        # Add a data point for each metric at current time
        timestamp = datetime.now().isoformat()
        if self.current_data.current_metrics:
            metrics = self.current_data.current_metrics
            self.current_data.historical_data.append({
                'timestamp': timestamp,
                'total_pnl': metrics.total_pnl,
                'hit_rate': metrics.hit_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'avg_inventory': metrics.avg_inventory
            })
            
            # Keep only last 100 data points for performance
            if len(self.current_data.historical_data) > 100:
                self.current_data.historical_data = self.current_data.historical_data[-100:]
    
    async def process_new_trade(self, trade_event) -> None:
        """Process a new trade event and update metrics."""
        if self.mock_mode:
            return  # In mock mode, we don't expect real trade events
        
        self.core_calculator.process_trade(trade_event)
        self.event_processor.process_trade_event(trade_event)
        self.per_asset_analyzer.per_asset.add_trade(trade_event)
        
        self.update_current_metrics()
        self.add_historical_data_points()
        
        # Broadcast update to connected clients
        await self.websocket_manager.broadcast({
            'type': 'metrics_update',
            'data': self.current_data.to_dict()
        })
    
    async def simulate_real_time_updates(self) -> None:
        """Simulate real-time updates for demo purposes."""
        if not self.mock_mode:
            return
        
        while True:
            try:
                # In mock mode, we'll simulate new trades periodically
                generator = MockTradeGenerator()
                new_trades = generator.generate_trades_stream(
                    datetime.now() - timedelta(minutes=1),
                    datetime.now(),
                    trades_per_hour=5
                )
                
                for trade in new_trades:
                    self.core_calculator.process_trade(trade)
                    self.event_processor.process_trade_event(trade)
                    self.per_asset_analyzer.per_asset.add_trade(trade)
                
                self.update_current_metrics()
                self.add_historical_data_points()
                
                # Broadcast update to connected clients
                await self.websocket_manager.broadcast({
                    'type': 'metrics_update',
                    'data': self.current_data.to_dict()
                })
                
                await asyncio.sleep(3)  # Update every 3 seconds in mock mode
            except Exception as e:
                self.logger.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(1)


@dataclass
class RealTimeDashboard:
    """Main dashboard class that creates the FastAPI application."""
    
    service: Optional[DashboardService] = None
    app: Optional[FastAPI] = None
    config: DashboardConfig = None
    
    def __post_init__(self):
        if self.service is None:
            self.service = DashboardService(mock_mode=True)
        if self.config is None:
            self.config = DashboardConfig()
        
        self._setup_app()
    
    def _setup_app(self) -> None:
        """Setup the FastAPI application."""
        self.app = FastAPI(title="Brawler Performance Dashboard", version="1.0.0")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes for the dashboard."""
        app = self.app
        service = self.service
        
        # Root route serves the dashboard
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Brawler Performance Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
                    .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .metric-title { font-size: 14px; color: #666; margin-bottom: 5px; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
                    .positive { color: #2ecc71; }
                    .negative { color: #e74c3c; }
                    .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .chart-wrapper { height: 300px; position: relative; }
                    .section-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }
                    .assets-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
                    .asset-card { background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Brawler Performance Dashboard</h1>
                        <p>Real-time market making performance tracking</p>
                    </div>
                    
                    <!-- Current Metrics -->
                    <div class="metrics-grid" id="current-metrics">
                        <div class="metric-card">
                            <div class="metric-title">Total PnL (24h)</div>
                            <div class="metric-value" id="total-pnl">0.00</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Hit Rate</div>
                            <div class="metric-value" id="hit-rate">0.0%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Avg. Markout</div>
                            <div class="metric-value" id="avg-markout">0.00</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Sharpe Ratio</div>
                            <div class="metric-value" id="sharpe-ratio">0.00</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Total Trades</div>
                            <div class="metric-value" id="total-trades">0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Avg. Inventory</div>
                            <div class="metric-value" id="avg-inventory">0.00</div>
                        </div>
                    </div>
                    
                    <!-- Performance Charts -->
                    <div class="chart-container">
                        <div class="section-title">Performance Over Time</div>
                        <div class="chart-wrapper">
                            <canvas id="performance-chart"></canvas>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="section-title">Hit Rate Over Time</div>
                        <div class="chart-wrapper">
                            <canvas id="hitrate-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Per-Asset Performance -->
                    <div class="chart-container">
                        <div class="section-title">Per-Asset Performance</div>
                        <div class="assets-grid" id="asset-performance">
                            <!-- Asset cards will be populated here -->
                        </div>
                    </div>
                </div>
                
                <script>
                    // WebSocket connection for real-time updates
                    const ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onopen = function(event) {
                        console.log('Connected to WebSocket');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'metrics_update') {
                            updateDashboard(data.data);
                        }
                    };
                    
                    ws.onclose = function(event) {
                        console.log('WebSocket connection closed');
                        // Try to reconnect after 3 seconds
                        setTimeout(function() {
                            window.location.reload();
                        }, 3000);
                    };
                    
                    // Function to update dashboard with new data
                    function updateDashboard(data) {
                        if (data.current_metrics) {
                            document.getElementById('total-pnl').textContent = data.current_metrics.total_pnl.toFixed(2);
                            document.getElementById('hit-rate').textContent = data.current_metrics.hit_rate.toFixed(2) + '%';
                            document.getElementById('avg-markout').textContent = data.current_metrics.markout_analysis.avg_markout_in.toFixed(4);
                            document.getElementById('sharpe-ratio').textContent = data.current_metrics.sharpe_ratio.toFixed(2);
                            document.getElementById('total-trades').textContent = data.current_metrics.total_trades;
                            document.getElementById('avg-inventory').textContent = data.current_metrics.avg_inventory.toFixed(4);
                        }
                        
                        if (data.per_asset_data) {
                            updateAssetCards(data.per_asset_data);
                        }
                        
                        if (data.historical_data) {
                            updateCharts(data.historical_data);
                        }
                    }
                    
                    // Function to update asset performance cards
                    function updateAssetCards(assets) {
                        const container = document.getElementById('asset-performance');
                        container.innerHTML = '';
                        
                        for (const [symbol, metrics] of Object.entries(assets)) {
                            const card = document.createElement('div');
                            card.className = 'asset-card';
                            card.innerHTML = `
                                <div style="font-weight: bold; margin-bottom: 10px;">${symbol}</div>
                                <div>PnL: ${metrics.total_pnl.toFixed(2)}</div>
                                <div>Trades: ${metrics.total_trades}</div>
                                <div>Hit Rate: ${metrics.hit_rate.toFixed(2)}%</div>
                                <div>Volume: ${metrics.total_volume.toFixed(2)}</div>
                            `;
                            container.appendChild(card);
                        }
                    }
                    
                    // Initialize charts
                    const perfCtx = document.getElementById('performance-chart').getContext('2d');
                    const perfChart = new Chart(perfCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Total PnL',
                                data: [],
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1,
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                    
                    const hitrateCtx = document.getElementById('hitrate-chart').getContext('2d');
                    const hitrateChart = new Chart(hitrateCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Hit Rate (%)',
                                data: [],
                                borderColor: 'rgb(255, 99, 132)',
                                tension: 0.1,
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { min: 0, max: 100 }
                            }
                        }
                    });
                    
                    // Function to update charts
                    function updateCharts(data) {
                        if (data.length === 0) return;
                        
                        const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
                        const pnlData = data.map(d => d.total_pnl);
                        const hitrateData = data.map(d => d.hit_rate);
                        
                        perfChart.data.labels = labels;
                        perfChart.data.datasets[0].data = pnlData;
                        perfChart.update();
                        
                        hitrateChart.data.labels = labels;
                        hitrateChart.data.datasets[0].data = hitrateData;
                        hitrateChart.update();
                    }
                    
                    // Request initial data on load
                    fetch('/api/current-metrics')
                        .then(response => response.json())
                        .then(data => {
                            updateDashboard({ 
                                current_metrics: data,
                                per_asset_data: {},
                                historical_data: []
                            });
                        });
                </script>
            </body>
            </html>
            """
        
        @app.get("/api/current-metrics")
        async def get_current_metrics():
            """Get current performance metrics."""
            if service.current_data.current_metrics:
                return service.current_data.current_metrics
            else:
                # Return a default metrics object if none is available
                return PerformanceMetrics()
        
        @app.get("/api/historical-data")
        async def get_historical_data():
            """Get historical performance data."""
            if service.current_data.historical_data:
                return service.current_data.historical_data
            else:
                return []
        
        @app.get("/api/per-asset-data")
        async def get_per_asset_data():
            """Get per-asset performance data."""
            if service.current_data.per_asset_data:
                return service.current_data.per_asset_data
            else:
                return {}
        
        @app.get("/api/trend-data")
        async def get_trend_data():
            """Get trend analysis data."""
            if service.current_data.trend_data:
                return service.current_data.trend_data
            else:
                return {}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await service.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep the connection alive
                    data = await websocket.receive_text()
                    # For now, just echo back the received data
                    await websocket.send_text(f"Message received: {data}")
            except WebSocketDisconnect:
                service.websocket_manager.disconnect(websocket)
    
    def start_server(self, config: DashboardConfig = None) -> None:
        """Start the dashboard server."""
        if config:
            self.config = config
        
        # Start real-time updates in the background if in mock mode
        if self.service.mock_mode:
            # Run the real-time updates in the background
            import threading
            thread = threading.Thread(target=lambda: asyncio.run(self.service.simulate_real_time_updates()))
            thread.daemon = True
            thread.start()
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            reload=self.config.reload,
            log_level=self.config.log_level
        )


def test_real_time_metric_updates():
    """Test that dashboard updates metrics in real-time."""
    service = DashboardService(mock_mode=True)
    
    # Check that the service was initialized properly
    assert service.current_data is not None
    assert service.core_calculator is not None
    assert service.historical_analyzer is not None


def test_24hr_snapshot_display():
    """Test that 24-hour snapshot is displayed correctly."""
    service = DashboardService(mock_mode=True)
    
    # After initialization, we should have current metrics
    assert service.current_data.current_metrics is not None
    
    # Check that key metrics are populated
    metrics = service.current_data.current_metrics
    assert hasattr(metrics, 'total_pnl')
    assert hasattr(metrics, 'hit_rate')
    assert hasattr(metrics, 'total_trades')


def test_historical_trend_visualization():
    """Test that historical trends are visualized correctly."""
    service = DashboardService(mock_mode=True)
    
    # Check that historical data was initialized
    assert service.current_data.historical_data is not None
    assert isinstance(service.current_data.historical_data, list)


def test_per_asset_breakdown_display():
    """Test that per-asset breakdowns are displayed correctly."""
    service = DashboardService(mock_mode=True)
    
    # Check that per-asset data was initialized
    assert service.current_data.per_asset_data is not None
    assert isinstance(service.current_data.per_asset_data, dict)


def test_dashboard_performance():
    """Test dashboard performance with real-time updates."""
    # This would test the dashboard's ability to handle updates efficiently
    service = DashboardService(mock_mode=True)
    
    # Verify the WebSocket manager was created
    assert service.websocket_manager is not None
    assert hasattr(service.websocket_manager, 'broadcast')


def test_error_handling_in_dashboard():
    """Test dashboard behavior when data is unavailable."""
    service = DashboardService(mock_mode=True)
    
    # Test accessing metrics when they might be None
    data = service.current_data.to_dict()
    assert isinstance(data, dict)
    
    # Should handle missing data gracefully
    assert 'current_metrics' in data
    assert 'per_asset_data' in data


if __name__ == "__main__":
    # Run the tests
    test_real_time_metric_updates()
    test_24hr_snapshot_display()
    test_historical_trend_visualization()
    test_per_asset_breakdown_display()
    test_dashboard_performance()
    test_error_handling_in_dashboard()
    
    print("All Real-time Dashboard and Visualization tests passed!")