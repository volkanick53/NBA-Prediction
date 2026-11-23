# Daily NBA prediction job (Cloud Run Job)
#   /app/data is mounted from a Cloud Storage bucket at runtime so the ESPN cache,
#   the BigQuery history cache and the JSON outputs survive between runs.
#   Credentials come from the job's service account - no key file in the image.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-job.txt .
RUN pip install --no-cache-dir -r requirements-job.txt

COPY engine/ engine/
COPY models/*.py models/
COPY models/game_model.json models/props_*.json models/

# default: today's slate (US Eastern date); Cloud Scheduler can override args
ENTRYPOINT ["python", "-m", "engine.live_engine", "--out-dir", "data/output"]
