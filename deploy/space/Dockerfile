# Hugging Face Space (Docker SDK): builds the API + Telegram bot from the public GitHub repo.
# To pick up new commits: Space settings -> "Factory reboot" (or change CACHEBUST below).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/home/user/app/.venv/bin:$PATH"
WORKDIR /home/user/app

ARG CACHEBUST=1
RUN git clone --depth 1 https://github.com/Gulshan-heap/niyam.git . \
    && uv sync --frozen --no-dev
COPY --chown=user start.sh ./start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]
