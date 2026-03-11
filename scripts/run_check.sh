#!/bin/bash
set -a
[ -f config/env/.brawler ] && . config/env/.brawler
[ -f config/env/.gradient ] && . config/env/.gradient
set +a
/home/ubuntu/slipstream/.venv/bin/python scripts/check_brawler_orders.py
