# Prod image (docs/40-docker-prod.md): the app only. oMLX and the ASR service stay on the host.
# Build with scripts/prod.sh build; run with scripts/prod.sh up (docker-compose.yml).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system safe.directory '*' \
 && useradd --uid 1000 --create-home app

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-docker.txt

COPY . .
# Browser libs, the Wanko model and the backdrops are gitignored: bake them in.
RUN bash scripts/fetch_assets.sh && chown -R app:app /app

USER app
EXPOSE 8010
CMD ["python", "chatbot.py"]
