"""
engine/bigquery_writer.py
==========================
Tahmin sonuçlarını BigQuery'e yazar.
Tablo mevcut değilse otomatik olarak oluşturur.

Tablo: nba-analytics-503718.nba_analytics.daily_predictions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

log = logging.getLogger(__name__)

KEY_PATH   = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"
TABLE_ID   = "daily_predictions"
FULL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# BigQuery şeması
BQ_SCHEMA = [
    bigquery.SchemaField("prediction_date",   "DATE"),
    bigquery.SchemaField("event_id",          "STRING"),
    bigquery.SchemaField("home_team_id",      "INTEGER"),
    bigquery.SchemaField("away_team_id",      "INTEGER"),
    bigquery.SchemaField("home_team_name",    "STRING"),
    bigquery.SchemaField("away_team_name",    "STRING"),
    bigquery.SchemaField("pred_home_score",   "FLOAT"),
    bigquery.SchemaField("pred_away_score",   "FLOAT"),
    bigquery.SchemaField("pred_spread",       "FLOAT"),
    bigquery.SchemaField("pred_total",        "FLOAT"),
    bigquery.SchemaField("home_win_prob",     "FLOAT"),
    bigquery.SchemaField("pred_game_pace",    "FLOAT"),
    bigquery.SchemaField("player_props_json", "STRING"),   # JSON string
    bigquery.SchemaField("created_at",        "TIMESTAMP"),
]


def _get_client() -> bigquery.Client:
    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    return bigquery.Client(credentials=creds, project=PROJECT_ID)


def _ensure_table(client: bigquery.Client) -> None:
    """Tablo yoksa BigQuery'de oluştur."""
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    try:
        client.get_table(table_ref)
        log.info("BigQuery tablosu mevcut: %s", FULL_TABLE)
    except Exception:
        log.info("BigQuery tablosu bulunamadı, oluşturuluyor: %s", FULL_TABLE)
        table = bigquery.Table(table_ref, schema=BQ_SCHEMA)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="prediction_date",
        )
        client.create_table(table)
        log.info("Tablo oluşturuldu: %s", FULL_TABLE)


def write_predictions(predictions: list[dict[str, Any]], date_str: str) -> None:
    """
    Tahmin listesini BigQuery tablosuna yaz.

    predictions: predict_daily.py'den dönen maç tahmin listesi.
    date_str: 'YYYYMMDD' formatında tarih.
    """
    if not predictions:
        log.warning("Yazılacak tahmin bulunamadı.")
        return

    client = _get_client()
    _ensure_table(client)

    # YYYYMMDD → DATE
    prediction_date = datetime.strptime(date_str, "%Y%m%d").date()
    now = datetime.utcnow()

    rows = []
    for p in predictions:
        rows.append({
            "prediction_date":   prediction_date.isoformat(),
            "event_id":          str(p.get("event_id", "")),
            "home_team_id":      int(p.get("home_team_id", 0)),
            "away_team_id":      int(p.get("away_team_id", 0)),
            "home_team_name":    str(p.get("home_team_name", "")),
            "away_team_name":    str(p.get("away_team_name", "")),
            "pred_home_score":   float(p.get("pred_home_score", 0)),
            "pred_away_score":   float(p.get("pred_away_score", 0)),
            "pred_spread":       float(p.get("pred_spread", 0)),
            "pred_total":        float(p.get("pred_total", 0)),
            "home_win_prob":     float(p.get("home_win_prob", 0.5)),
            "pred_game_pace":    float(p.get("pred_game_pace", 0)),
            "player_props_json": json.dumps(p.get("player_props", [])),
            "created_at":        now.isoformat(),
        })

    errors = client.insert_rows_json(FULL_TABLE, rows)
    if errors:
        log.error("BigQuery insert hataları: %s", errors)
        raise RuntimeError(f"BigQuery insert başarısız: {errors}")
    log.info("%d maç tahmini BigQuery'e yazıldı.", len(rows))
