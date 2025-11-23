# SLIPSTREAM TRADING PLATFORM - ONBOARDING GUIDE

Welcome to the Slipstream Federated Trading Platform! This guide will get you up to speed on how to run existing strategies, analyze their performance, and build new ones.

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [System Architecture](#system-architecture) 
3. [Getting Started](#getting-started)
4. [Running Existing Strategies](#running-existing-strategies)
5. [Analyzing Performance](#analyzing-performance)
6. [Building New Strategies](#building-new-strategies)
7. [Advanced Topics](#advanced-topics)

## Platform Overview

Slipstream is a federated trading factory that transforms the team from "traders" to "capital allocators." The platform operates using the **USA Model** architecture where:
- Individual strategies ("Strategy Pods") operate as autonomous units
- The central system ("Meta-Optimizer") manages budget and quality assessment
- Shared infrastructure ("Grid") provides common services

### Key Principles
- **Allocation over Orchestration**: The central system tells strategies how much risk to take, not what to buy
- **Federated Architecture**: Strategies maintain autonomy while being coordinated through risk allocation
- **Continuous Improvement**: Kaizen-enabled through comprehensive pre/post-trade analysis
- **Modular Design**: Easy addition of new strategies without disrupting existing operations

## System Architecture

### Layer 1: Strategy Pods (The "States")
- **Self-Contained Services**: Each strategy is a black box with autonomous operation
- **Standardized Interface**: All strategies expose three required endpoints:
  - `GET /status`: Returns Net Exposure, Open Orders, and PnL
  - `POST /configure`: Accepts dynamic limits (Max Position, Volatility Target) 
  - `POST /halt`: Emergency kill-switch
- **Autonomy**: Strategies handle their own market data, signal generation, and order placement

### Layer 2: Meta-Optimizer (The "Central Bank")
- **Statistical Asset Management**: Views strategies as statistical assets, not code
- **Allocation Cycle**: Every 4 hours (or daily), ingests performance data, calculates covariance, re-optimizes capital allocation
- **Performance-based Capital Flow**: High-performing strategies get more capital; low-performing strategies get "strangled"

### Layer 3: Shared Infrastructure (The "Grid")
- **Data Lake**: Single source of truth for OHLCV and Trade data
- **Execution Gateway**: Unified connector for exchange APIs
- **Risk Auditor**: Independent verification of strategy-reported exposures

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment tool (uv, pipenv, or venv)
- Exchange API keys (read-only for risk auditor)
- Git access to the repository

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/company/slipstream.git
cd slipstream

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or using uv (recommended):
uv venv
source .venv/bin/activate  
uv sync
```

### Configuration
Create your configuration files:
```bash
# Copy example configuration
cp config/gradient_live.json config/my_strategy.json
cp config/brawler_single_asset.example.yml config/my_brawler_config.yml
```

Edit configuration files with your specific parameters including exchange keys, risk limits, and strategy parameters.

### Environment Variables
```bash
# Create .env file with your credentials
echo "HYPERLIQUID_API_KEY=your_key_here" >> .env
echo "HYPERLIQUID_API_SECRET=your_secret_here" >> .env
```

## Running Existing Strategies

### Available Strategies
- **Gradient Strategy**: Balanced trend-following overlay built on shared tooling
- **Brawler Passive Market Maker**: CEX-anchored, volatility-aware quoting loop
- **Template Strategy**: Reference implementation for new strategy development
- **Volume Generator**: Volume generation utility (for testing only)

### Running the Gradient Strategy
```bash
# Run the gradient strategy with your config
uv run python -m slipstream.strategies.gradient.cli run_backtest_cli --returns-csv data/returns.csv --top-n 5

# Or for live trading:
uv run python -m slipstream.strategies.gradient.cli run_backtest_cli --config config/gradient_live.json

# Calculate gradient signals
uv run python -m slipstream.strategies.gradient.cli compute_signals_cli --returns-csv data/returns.csv --output data/signals/gradient_signals.csv
```

### Running the Brawler Strategy
```bash
# Run brawler strategy
uv run python -m slipstream.strategies.brawler.cli run_brawler_cli --config config/brawler_config.yml --log-level INFO

# Run brawler backtest
uv run python -m slipstream.strategies.brawler.cli run_backtest_cli --config config/brawler_config.yml
```

### Running with the Strategy Runner
The platform provides a unified interface for running strategies:

```bash
# List available strategies
python scripts/strategies/run_backtest.py --list-strategies

# Run any strategy
python scripts/strategies/run_backtest.py --strategy gradient -- --returns-csv data/returns.csv --top-n 5

# For live trading:
python scripts/strategies/run_live.py --strategy gradient --config config/my_config.json
```

### Monitoring Running Strategies
```bash
# Check strategy status via API
curl http://localhost:8000/status

# Get strategy configuration
curl http://localhost:8000/config

# Force strategy halt
curl -X POST http://localhost:8000/halt
```

## Analyzing Performance

### Centralized Performance Dashboard
The federation provides centralized performance tracking:

#### Meta-Optimizer Dashboard
```bash
# Run the meta-optimizer performance tracker
python scripts/federation/performance_tracker.py
```

This provides:
- **Strategy Performance Metrics**: Sharpe ratio, volatility, drawdown
- **Correlation Analysis**: Portfolio correlation to benchmarks (BTC/ETH)
- **Capital Allocation**: Current allocation and recommended changes
- **Risk Metrics**: Volatility, VaR, and stress-test results

#### Audit and Compliance
```bash
# Audit strategy exposures against actual trading
python scripts/federation/risk_auditor.py --strategy-id gradient_strategy_1

# Generate compliance reports
python scripts/federation/compliance_report.py --date-range 2024-01-01:2024-12-31
```

### Post-Trade Analysis
The system performs comprehensive execution quality analysis:

#### Implementation Shortfall Analysis
```bash
# Analyze execution quality
python scripts/post_trade/shortfall_analysis.py --strategy-id my_strategy --date-range 2024-01-01:2024-01-31

# Check for high shortfall trades (indicates execution problems vs. alpha issues)
python scripts/post_trade/high_shortfall_finder.py --strategy-id my_strategy
```

#### Residual Return Analysis
```bash
# Separate skill-based alpha from market-driven returns
python scripts/post_trade/residual_alpha.py --strategy-id my_strategy --benchmark BTC,ETH

# Detect fake alpha (leveraged market exposure)
python scripts/post_trade/fake_alpha_detector.py --strategy-id my_strategy
```

### Performance Attribution
```bash
# Detailed PnL attribution
python scripts/analysis/pnl_attribution.py --strategy-id my_strategy --period daily

# Risk factor attribution
python scripts/analysis/risk_attribution.py --strategy-id my_strategy
```

### Custom Analysis
You can create custom analysis scripts by importing the analysis modules:

```python
from slipstream.federation.optimizer import MetaOptimizer
from slipstream.federation.auditor import RiskAuditor
from slipstream.federation.shortfall import ImplementationShortfallAnalyzer

# Your custom analysis code here
```

## Building New Strategies

### Strategy Template Overview
All strategies follow the same template structure:

```
src/slipstream/strategies/
├── my_new_strategy/
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Configuration management  
│   ├── signals.py      # Signal generation
│   ├── portfolio.py    # Portfolio management
│   ├── execution.py    # Order execution
│   └── backtest.py     # Backtesting logic
```

### Step 1: Create Strategy Template

```bash
# Copy the template strategy
cp -r src/slipstream/strategies/template src/slipstream/strategies/my_strategy
```

### Step 2: Define Strategy API Contract

Every strategy must implement the standardized endpoints:

```python
# In src/slipstream/strategies/my_strategy/api.py

class MyStrategyAPI(StrategyAPI):
    def __init__(self, strategy_instance):
        super().__init__(strategy_instance)
        
    async def get_status(self) -> StrategyStatus:
        """Return current strategy status: Net Exposure, Open Orders, PnL."""
        return StrategyStatus(
            net_exposure=await self.strategy.get_net_exposure(),
            open_orders=await self.strategy.get_open_orders(),
            pnl=await self.strategy.get_pnl(),
            health_status="healthy",
            uptime=self._uptime,
            strategy_name="my_strategy"
        )
    
    async def configure(self, config_update: ConfigurationUpdate) -> Dict[str, Any]:
        """Accept dynamic limits configuration."""
        # Update strategy parameters based on config_update
        await self.strategy.update_configuration(config_update)
        return {"status": "success", "updated_params": config_update}
    
    async def halt(self, reason: Optional[HaltReason] = None, message: Optional[str] = None) -> Dict[str, Any]:
        """Emergency shutdown."""
        await self.strategy.stop_gracefully()
        return {"status": "halted", "reason": reason, "message": message}
```

### Step 3: Implement Core Strategy Logic

```python
# In src/slipstream/strategies/my_strategy/core.py

class MyStrategy:
    def __init__(self, config):
        self.config = config
        self.position_manager = PositionManager()
        self.signal_generator = MySignalGenerator()
        
    async def generate_signals(self):
        """Generate trading signals based on your strategy logic."""
        # Implement your signal generation logic
        return await self.signal_generator.generate()
        
    async def execute_orders(self, signals):
        """Execute orders based on signals."""
        # Implement your execution logic
        return await self.position_manager.execute(signals)
        
    async def get_pnl(self):
        """Return current PnL."""
        return await self.position_manager.calculate_pnl()
```

### Step 4: Add Configuration Management

```python
# In src/slipstream/strategies/my_strategy/config.py

@dataclass
class MyStrategyConfig:
    symbol: str = "BTC"
    capital: float = 10000.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 5000.0
    volatility_target: float = 0.15
    lookback_period: int = 24  # Hours
```

### Step 5: Create Command-Line Interface

```python
# In src/slipstream/strategies/my_strategy/cli.py

def run_my_strategy_cli(argv: Optional[Iterable[str]] = None) -> None:
    """Command-line interface for my strategy."""
    parser = argparse.ArgumentParser(description="Run my strategy")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    
    args = parser.parse_args(argv)
    
    # Load configuration
    config = load_my_strategy_config(args.config)
    
    # Initialize and run strategy
    strategy = MyStrategy(config)
    strategy.run(paper_mode=args.paper)
```

### Step 6: Add Backtesting Capability

```python
# In src/slipstream/strategies/my_strategy/backtest.py

def run_my_strategy_backtest(config, historical_data):
    """Run backtest for my strategy."""
    strategy = MyStrategy(config)
    
    results = []
    for time_step in historical_data:
        signals = strategy.generate_signals(time_step)
        pnl = strategy.execute_orders_with_data(signals, time_step)
        results.append({
            'timestamp': time_step.timestamp,
            'pnl': pnl,
            'position': strategy.get_current_position(),
            'signals': signals
        })
    
    return BacktestResults(results)
```

### Step 7: Register the Strategy

Add your strategy to the registry in `src/slipstream/strategies/__init__.py`:

```python
STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {
    # ... existing strategies ...
    "my_strategy": StrategyInfo(
        key="my_strategy",
        title="My New Strategy", 
        module="slipstream.strategies.my_strategy",
        description="Description of your strategy.",
        cli_entrypoints={
            "run_backtest": "slipstream.strategies.my_strategy.cli.run_backtest_cli",
            "run_live": "slipstream.strategies.my_strategy.cli.run_strategy_cli",
        },
    ),
}
```

### Step 8: Test Your Strategy

```bash
# Run unit tests
python -m pytest tests/strategies/test_my_strategy.py

# Run backtest
python scripts/strategies/run_backtest.py --strategy my_strategy -- --config config/my_strategy.json

# For development, run with limited data
python scripts/strategies/run_backtest.py --strategy my_strategy -- --config config/my_strategy.json --days 30
```

### Best Practices for Strategy Development

1. **Follow the API Contract**: Your strategy must implement the standardized endpoints
2. **Log Everything**: Comprehensive logging for debugging and analysis
3. **Handle Errors Gracefully**: Implement proper error handling and recovery
4. **Manage Risk**: Implement position limits and stop losses
5. **Test Thoroughly**: Run backtests with various market conditions
6. **Document**: Comment your code and strategy logic clearly

## Advanced Topics

### Custom Risk Management

```python
from slipstream.federation.risk import RiskManager

class MyCustomRiskManager(RiskManager):
    def __init__(self, strategy_id):
        super().__init__(strategy_id)
        
    async def assess_trade_risk(self, trade_request):
        """Custom risk assessment before executing trade."""
        # Implement your custom risk logic
        return await self.validate_against_portfolio_limits(trade_request)
```

### Portfolio Integration

Strategies can participate in the broader portfolio optimization:

```python
from slipstream.federation.optimizer import MetaOptimizer

# Get recommendations based on portfolio correlation
optimizer = MetaOptimizer()
portfolio_recommendation = await optimizer.get_position_recommendation(
    strategy_id="my_strategy",
    base_position=5000.0,
    correlation_matrix=await optimizer.get_correlation_matrix()
)
```

### Monitoring and Alerting

Set up custom alerts for your strategy:

```python
from slipstream.core.notifications import AlertManager

alert_manager = AlertManager()
await alert_manager.subscribe_to_events(
    strategy_id="my_strategy",
    events=["high_drawdown", "low_sharpe", "execution_failure"],
    channels=["email", "slack"]
)
```

### Performance Optimization

To optimize strategy performance:
1. **Reduce Latency**: Optimize data processing and execution paths
2. **Efficient Data Structures**: Use pandas/numpy for numerical computations
3. **Async Operations**: Use async/await for I/O operations
4. **Caching**: Cache expensive calculations when appropriate

### Debugging Strategies

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m slipstream.strategies.my_strategy.cli run_strategy --config my_config.json

# Monitor in real-time
tail -f logs/my_strategy.log

# Check federation health
curl http://localhost:8080/federation/health
```

## Troubleshooting

### Common Issues

1. **Strategy not responding to API calls**: Check that the strategy implements the required endpoints correctly
2. **Poor backtest performance**: Verify your signal generation and execution logic
3. **Risk limits exceeded**: Review position sizing and risk management code
4. **Connection issues**: Verify API keys and network connectivity

### Performance Issues

- Monitor CPU/memory usage: `htop` or `top`
- Check for memory leaks in long-running processes
- Profile slow functions: `python -m cProfile script.py`

### Data Issues

- Ensure OHLCV data is complete and accurate
- Check for missing dates or gaps in historical data
- Verify data timezone consistency

## Resources and Support

### Documentation
- [Main Documentation](docs/DOCUMENTATION.md) - Complete technical reference
- [Federated Vision](FEDERATED_VISION.md) - Architecture and design principles
- [Strategy Templates](docs/STRATEGY_DESIGN.md) - Strategy development guidelines
- [Risk Management](docs/RISK_MANAGEMENT.md) - Risk policies and procedures

### Getting Help
- Create a GitHub issue for technical problems
- Reach out to the development team on Slack
- Review existing strategies for implementation examples
- Check the troubleshooting section above

### Contributing
- Follow the existing code style and patterns
- Write comprehensive tests for new features
- Update documentation when adding new functionality
- Follow the TDD approach outlined in the migration plan

---

**Next Steps:**
1. Set up your development environment
2. Run the Gradient strategy backtest to understand the system
3. Explore the backtesting results and analysis tools
4. Build your first simple strategy using the template
5. Test thoroughly before deploying to production

Happy trading!