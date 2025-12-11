#!/bin/bash
set -a
[ -f .env.brawler ] && . .env.brawler
[ -f .env.gradient ] && . .env.gradient
set +a
/home/ubuntu/slipstream/.venv/bin/python scripts/check_brawler_orders.py
