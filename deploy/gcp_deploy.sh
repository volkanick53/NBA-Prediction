#!/usr/bin/env bash
# =============================================================================
# Deploys the daily NBA prediction engine to Google Cloud:
#   Cloud Run Job  (container built from ./Dockerfile by Cloud Build)
#   Cloud Storage  bucket mounted at /app/data  (ESPN cache, BigQuery cache, JSON outputs)
#   Cloud Scheduler  2 runs per day, US Eastern time:
#       10:00 ET  (TR 18:00)  first prediction of the day's slate
#       17:30 ET  (TR 01:30)  refresh with the latest injury report before tip-off
#   Predictions -> BigQuery nba_analytics.pred_games / pred_player_props
#
# Safe to re-run: existing resources are kept, the job image is updated.
# Requires: gcloud CLI logged in as a project owner (gcloud auth login).
# Run from the repository root:   bash deploy/gcp_deploy.sh
# =============================================================================
set -euo pipefail
# Git Bash on Windows rewrites Unix-looking paths in arguments; keep the
# container mount path (/app/data) untouched without breaking gcloud itself.
export MSYS2_ARG_CONV_EXCL="volume="

PROJECT=nba-analytics-503718
REGION=europe-west1                      # BigQuery dataset is in the EU
BUCKET=${PROJECT}-nba-engine
SA_NAME=nba-predictor
SA=${SA_NAME}@${PROJECT}.iam.gserviceaccount.com
JOB=nba-daily-predictions
RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"

gcloud config set project "$PROJECT" >/dev/null

echo "== 1/6 APIs"
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com \
  bigquery.googleapis.com

echo "== 2/6 Service account (least privilege)"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="NBA daily prediction job"
for ROLE in roles/bigquery.dataEditor roles/bigquery.jobUser; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
    --role="$ROLE" --condition=None >/dev/null
done

echo "== 3/6 Storage bucket"
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" \
  --role=roles/storage.objectAdmin >/dev/null
# seed caches from this machine (saves ~300 ESPN requests on the first cloud run)
[ -d data/espn_cache ] && gcloud storage rsync data/espn_cache "gs://$BUCKET/espn_cache" --quiet
[ -d data/cache ] && gcloud storage rsync data/cache "gs://$BUCKET/cache" --quiet

echo "== 4/6 Cloud Run Job (build + deploy)"
gcloud run jobs deploy "$JOB" --source . --region "$REGION" \
  --service-account "$SA" --memory 2Gi --cpu 1 --task-timeout 20m --max-retries 1 \
  --add-volume name=data,type=cloud-storage,bucket="$BUCKET" \
  --add-volume-mount volume=data,mount-path=/app/data

echo "== 5/6 Scheduler may start the job"
gcloud run jobs add-iam-policy-binding "$JOB" --region "$REGION" \
  --member="serviceAccount:$SA" --role=roles/run.invoker >/dev/null

echo "== 6/6 Schedules (America/New_York)"
schedule() {   # name cron
  if gcloud scheduler jobs describe "$1" --location "$REGION" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "$1" --location "$REGION" --schedule "$2" \
      --time-zone "America/New_York" --uri "$RUN_URI" --http-method POST \
      --oauth-service-account-email "$SA"
  else
    gcloud scheduler jobs create http "$1" --location "$REGION" --schedule "$2" \
      --time-zone "America/New_York" --uri "$RUN_URI" --http-method POST \
      --oauth-service-account-email "$SA"
  fi
}
schedule nba-predictions-morning "0 10 * * *"
schedule nba-predictions-pretip  "30 17 * * *"

echo
echo "Done. Run once now:   gcloud run jobs execute $JOB --region $REGION --wait"
echo "Logs:                 gcloud run jobs executions list --job $JOB --region $REGION"
