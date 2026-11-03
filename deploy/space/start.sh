#!/usr/bin/env bash
# Runs the API (port 7860, what Spaces expose) and, when a token is set, the Telegram bot.
set -euo pipefail

alembic upgrade head

uvicorn niyam.api.main:app --host 0.0.0.0 --port 7860 &
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  python -m niyam.bot.telegram &
fi

# Exit when either process dies so Spaces restarts the container.
wait -n
