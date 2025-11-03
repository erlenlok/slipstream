"""Notification system for live trading alerts."""

import os
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError


async def send_telegram_rebalance_alert(
    rebalance_data: Dict[str, Any],
    config: Any
) -> bool:
    """
    Send Telegram alert after rebalance completes.

    Args:
        rebalance_data: Dictionary with rebalance summary
        config: Configuration object

    Returns:
        True if sent successfully, False otherwise
    """
    if not config.alerts_enabled:
        return False

    token = os.getenv("TELEGRAM_BOT_TOKEN", config.telegram_token)
    chat_id = os.getenv("TELEGRAM_CHAT_ID", config.telegram_chat_id)

    if not token or not chat_id:
        print("Warning: Telegram credentials not configured")
        return False

    try:
        bot = Bot(token=token)

        # Format message
        timestamp = rebalance_data.get("timestamp", datetime.now().isoformat())
        n_long = rebalance_data.get("n_long", 0)
        n_short = rebalance_data.get("n_short", 0)
        turnover = rebalance_data.get("total_turnover", 0)
        stage1_filled = rebalance_data.get("stage1_filled", 0)
        stage2_filled = rebalance_data.get("stage2_filled", 0)
        errors = rebalance_data.get("errors", 0)
        dry_run = rebalance_data.get("dry_run", True)
        total_orders = rebalance_data.get("total_orders", stage1_filled + stage2_filled)
        passive_rate = rebalance_data.get("passive_fill_rate")
        aggressive_rate = rebalance_data.get("aggressive_fill_rate")
        passive_slip = rebalance_data.get("passive_slippage") or {}
        aggressive_slip = rebalance_data.get("aggressive_slippage") or {}
        total_slip = rebalance_data.get("total_slippage") or {}
        stage2_highlights = rebalance_data.get("stage2_highlights") or []
        signal_tracking = rebalance_data.get("signal_tracking") or {}
        tracking_metrics = signal_tracking.get("metrics") or {}

        status_emoji = "✅" if errors == 0 else "⚠️"
        mode_tag = "[DRY-RUN]" if dry_run else "[LIVE]"

        def format_rate(filled: int, rate: Optional[float]) -> str:
            base = f"{filled}/{total_orders}" if total_orders else f"{filled}"
            return f"{base} ({rate:.1%})" if rate is not None else base

        def format_slippage(stats: Dict[str, Any]) -> str:
            notional = stats.get("total_usd")
            bps = stats.get("weighted_bps")
            if notional is None or bps is None:
                return "n/a"
            return f"{bps:+.2f} bps on ${notional:,.0f}"

        fallback_lines = ""
        if stage2_highlights:
            formatted_lines = []
            for item in stage2_highlights:
                asset = item.get("asset", "N/A")
                notional = item.get("notional_usd", 0.0)
                slip = item.get("slippage_bps")
                slip_str = "n/a" if slip is None else f"{slip:+.2f} bps"
                formatted_lines.append(f"  • {asset}: ${notional:,.0f} at {slip_str}")
            fallback_lines = "\n🚨 **Fallback Fills**:\n" + "\n".join(formatted_lines)
        signal_lines = ""
        if signal_tracking and tracking_metrics:
            forecast_ts = signal_tracking.get("forecast_timestamp", "n/a")
            corr = tracking_metrics.get("pearson_corr")
            hit = tracking_metrics.get("hit_rate")
            mae = tracking_metrics.get("mae")
            rmse = tracking_metrics.get("rmse")

            corr_str = f"{corr:.3f}" if isinstance(corr, (int, float)) else "n/a"
            hit_str = f"{hit:.1%}" if isinstance(hit, (int, float)) else "n/a"
            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else "n/a"
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "n/a"

            signal_lines = (
                "\n📈 **Signal Tracking**:\n"
                f"  • Forecast @ {forecast_ts}\n"
                f"  • Corr: {corr_str} | Hit: {hit_str}\n"
                f"  • MAE: {mae_str} | RMSE: {rmse_str}"
            )

        message = f"""
{status_emoji} **Gradient Rebalance Complete** {mode_tag}

📅 **Time**: {timestamp}

📊 **Positions**:
  • Long: {n_long} positions
  • Short: {n_short} positions
  • Total: {n_long + n_short}

💰 **Execution**:
  • Turnover: ${turnover:,.2f}
  • Stage 1 (passive): {format_rate(stage1_filled, passive_rate)}
  • Stage 2 (aggressive): {format_rate(stage2_filled, aggressive_rate)}
  • Errors: {errors}

📉 **Slippage (bps | $)**:
  • Stage 1: {format_slippage(passive_slip)}
  • Stage 2: {format_slippage(aggressive_slip)}
  • Total: {format_slippage(total_slip)}

{signal_lines}
{fallback_lines}
{'⚠️ DRY-RUN MODE - No real orders placed' if dry_run else '✅ Live positions entered'}
"""

        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown"
        )

        print(f"✓ Telegram alert sent to chat {chat_id}")
        return True

    except TelegramError as e:
        print(f"Error sending Telegram message: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error sending Telegram: {e}")
        return False


def send_telegram_rebalance_alert_sync(
    rebalance_data: Dict[str, Any],
    config: Any
) -> bool:
    """Synchronous wrapper for send_telegram_rebalance_alert."""
    try:
        return asyncio.run(send_telegram_rebalance_alert(rebalance_data, config))
    except Exception as e:
        print(f"Error in Telegram sync wrapper: {e}")
        return False


def send_email_daily_summary(
    daily_summary: Dict[str, Any],
    all_time_summary: Dict[str, Any],
    config: Any
) -> bool:
    """
    Send daily performance summary via email.

    Args:
        daily_summary: Daily performance metrics
        all_time_summary: All-time aggregated metrics
        config: Configuration object

    Returns:
        True if sent successfully, False otherwise
    """
    if not config.alerts_enabled:
        return False

    # Get email configuration from environment
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    email_from = os.getenv("EMAIL_FROM", "")
    email_to = os.getenv("EMAIL_TO", "")
    email_password = os.getenv("EMAIL_PASSWORD", "")

    if not email_from or not email_to or not email_password:
        print("Warning: Email credentials not configured")
        return False

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Gradient Daily Summary - {daily_summary.get('date', 'N/A')}"
        msg["From"] = email_from
        msg["To"] = email_to

        # Create HTML body
        html_body = create_daily_summary_html(daily_summary, all_time_summary)

        # Attach HTML
        html_part = MIMEText(html_body, "html")
        msg.attach(html_part)

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)

        print(f"✓ Daily summary email sent to {email_to}")
        return True

    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def create_daily_summary_html(
    daily: Dict[str, Any],
    all_time: Dict[str, Any]
) -> str:
    """Create HTML email body for daily summary."""

    date = daily.get("date", "N/A")
    n_rebalances = daily.get("n_rebalances", 0)

    if n_rebalances == 0:
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
        <h2>Gradient Daily Summary - {date}</h2>
        <p><strong>No rebalances occurred on this day.</strong></p>
        </body>
        </html>
        """

    # Daily metrics
    total_turnover = daily.get("total_turnover", 0)
    avg_turnover = daily.get("avg_turnover_per_rebalance", 0)
    avg_positions = daily.get("avg_positions", 0)
    avg_gross = daily.get("avg_gross_exposure", 0)
    avg_net = daily.get("avg_net_exposure", 0)
    passive_fills = daily.get("passive_fills", 0)
    aggressive_fills = daily.get("aggressive_fills", 0)
    passive_rate = daily.get("passive_fill_rate", 0) * 100
    aggressive_rate = daily.get("aggressive_fill_rate", 0) * 100
    errors = daily.get("total_errors", 0)

    # All-time metrics
    total_rebalances = all_time.get("total_rebalances", 0)
    days_running = all_time.get("days_running", 0)
    lifetime_turnover = all_time.get("total_turnover", 0)
    lifetime_passive_rate = all_time.get("passive_fill_rate", 0) * 100
    lifetime_errors = all_time.get("total_errors", 0)

    # Recent rebalances table
    rebalances = daily.get("rebalances", [])
    rebalance_rows = ""
    for rb in rebalances[-5:]:  # Show last 5
        timestamp = rb.get("timestamp", "N/A")
        n_pos = rb.get("n_positions", 0)
        turnover = rb.get("total_turnover", 0)
        s1 = rb.get("stage1_filled", 0)
        s2 = rb.get("stage2_filled", 0)
        err = rb.get("errors", 0)

        rebalance_rows += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{n_pos}</td>
            <td>${turnover:,.0f}</td>
            <td>{s1}</td>
            <td>{s2}</td>
            <td>{'❌' if err > 0 else '✅'}</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #2c3e50; }}
            h3 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .section {{ margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h2>📊 Gradient Daily Summary - {date}</h2>

        <div class="section">
            <h3>Today's Performance</h3>
            <div class="metric-box">
                <div class="metric-label">Rebalances</div>
                <div class="metric-value">{n_rebalances}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Turnover</div>
                <div class="metric-value">${total_turnover:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Avg Positions</div>
                <div class="metric-value">{avg_positions:.1f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Passive Fill Rate</div>
                <div class="metric-value">{passive_rate:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Errors</div>
                <div class="metric-value" style="color: {'#e74c3c' if errors > 0 else '#27ae60'};">{errors}</div>
            </div>
        </div>

        <div class="section">
            <h3>Execution Details</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Turnover per Rebalance</td>
                    <td>${avg_turnover:,.2f}</td>
                </tr>
                <tr>
                    <td>Average Gross Exposure</td>
                    <td>${avg_gross:,.2f}</td>
                </tr>
                <tr>
                    <td>Average Net Exposure</td>
                    <td>${avg_net:,.2f}</td>
                </tr>
                <tr>
                    <td>Passive Fills (Stage 1)</td>
                    <td>{passive_fills} ({passive_rate:.1f}%)</td>
                </tr>
                <tr>
                    <td>Aggressive Fills (Stage 2)</td>
                    <td>{aggressive_fills} ({aggressive_rate:.1f}%)</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h3>Recent Rebalances</h3>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Positions</th>
                    <th>Turnover</th>
                    <th>Stage 1</th>
                    <th>Stage 2</th>
                    <th>Status</th>
                </tr>
                {rebalance_rows}
            </table>
        </div>

        <div class="section">
            <h3>All-Time Statistics</h3>
            <div class="metric-box">
                <div class="metric-label">Days Running</div>
                <div class="metric-value">{days_running}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Rebalances</div>
                <div class="metric-value">{total_rebalances}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Lifetime Turnover</div>
                <div class="metric-value">${lifetime_turnover:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Lifetime Passive Rate</div>
                <div class="metric-value">{lifetime_passive_rate:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Lifetime Errors</div>
                <div class="metric-value">{lifetime_errors}</div>
            </div>
        </div>

        <hr style="margin-top: 40px;">
        <p style="color: #7f8c8d; font-size: 12px;">
            Generated by Gradient Live Trading System<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
        </p>
    </body>
    </html>
    """

    return html
