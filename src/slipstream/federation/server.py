"""
API server for federated strategy endpoints.

This module implements the HTTP server that exposes the standardized
API endpoints for strategy pods using FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .api import StrategyAPI, StrategyStatus, ConfigurationUpdate, HaltReason, wrap_strategy_for_api


class ConfigureRequest(BaseModel):
    """Request model for the configure endpoint."""
    max_position: Optional[float] = None
    volatility_target: Optional[float] = None
    risk_limits: Optional[Dict[str, Any]] = None
    other_params: Optional[Dict[str, Any]] = None


class HaltRequest(BaseModel):
    """Request model for the halt endpoint."""
    reason: Optional[str] = None
    message: Optional[str] = None


class StrategyAPIServer:
    """
    HTTP server that exposes standardized API endpoints for strategy pods.
    
    This server can be attached to any strategy to provide the required
    federation endpoints without modifying the strategy's core logic.
    """

    def __init__(self, strategy_api: StrategyAPI, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the API server.
        
        Args:
            strategy_api: The StrategyAPI wrapper for the target strategy
            host: Host to bind the server to
            port: Port to run the server on
        """
        self.strategy_api = strategy_api
        self.host = host
        self.port = port
        self.app = FastAPI(
            title=f"Federated Strategy API - {type(strategy_api.strategy).__name__}",
            description="Standardized API endpoints for federated strategy pods",
            version="1.0.0"
        )
        self.server_task: Optional[asyncio.Task] = None
        self._setup_routes()
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Set up the API routes according to the federation specification."""
        
        @self.app.get("/status", response_model=StrategyStatus)
        async def get_status() -> StrategyStatus:
            """GET /status: Returns Net Exposure, Open Orders, and PnL."""
            try:
                status = await self.strategy_api.get_status()
                await self.strategy_api.update_uptime()
                return status
            except Exception as e:
                self.logger.error(f"Error getting strategy status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/configure")
        async def configure(request: ConfigureRequest) -> Dict[str, Any]:
            """POST /configure: Accepts dynamic limits (Max Position, Volatility Target)."""
            try:
                config_update = ConfigurationUpdate(
                    max_position=request.max_position,
                    volatility_target=request.volatility_target,
                    risk_limits=request.risk_limits,
                    other_params=request.other_params
                )
                result = await self.strategy_api.configure(config_update)
                return result
            except Exception as e:
                self.logger.error(f"Error configuring strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/halt")
        async def halt(request: HaltRequest) -> Dict[str, Any]:
            """POST /halt: A hard kill-switch for emergency shutdown."""
            try:
                reason_enum = None
                if request.reason:
                    try:
                        reason_enum = HaltReason(request.reason.lower())
                    except ValueError:
                        self.logger.warning(f"Invalid halt reason: {request.reason}, using MANUAL")
                        reason_enum = HaltReason.MANUAL
                        
                result = await self.strategy_api.halt(
                    reason=reason_enum,
                    message=request.message
                )
                return result
            except Exception as e:
                self.logger.error(f"Error halting strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            """GET /health: Health check endpoint."""
            try:
                health_status = await self.strategy_api.heartbeat()
                return health_status
            except Exception as e:
                self.logger.error(f"Error checking strategy health: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/")
        async def root() -> Dict[str, str]:
            """Root endpoint providing API information."""
            strategy_name = type(self.strategy_api.strategy).__name__
            return {
                "message": f"Federated Strategy API for {strategy_name}",
                "endpoints": ["/status", "/configure", "/halt", "/health"],
                "status": "running"
            }

    async def start(self):
        """Start the API server."""
        import uvicorn
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        
        # Run the server in a task
        self.server_task = asyncio.create_task(server.serve())
        self.logger.info(f"Strategy API server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the API server."""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Strategy API server stopped")


def create_strategy_server(strategy_instance, host: str = "0.0.0.0", port: int = 8000) -> StrategyAPIServer:
    """
    Convenience function to create an API server for a strategy instance.
    
    Args:
        strategy_instance: The strategy instance to wrap with API
        host: Host to bind to
        port: Port to run on
        
    Returns:
        A StrategyAPIServer instance ready to start
    """
    strategy_api = wrap_strategy_for_api(strategy_instance)
    return StrategyAPIServer(strategy_api, host, port)


__all__ = [
    "StrategyAPIServer",
    "create_strategy_server",
    "ConfigureRequest",
    "HaltRequest",
]