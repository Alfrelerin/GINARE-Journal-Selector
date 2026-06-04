"""Calcula la mediana de tiempo desde envío hasta publicación online por
revista, usando las fechas que CrossRef guarda para cada DOI.

CrossRef expone para cada artículo:
  - received     (fecha de envío del manuscrito por el autor)
  - accepted     (fecha de aceptación tras revisión, no siempre presente)
  - published-online o published-print

`time_to_publication_weeks` = mediana de (published − received) en semanas.
`time_to_acceptance_weeks`  = mediana de (accepted − received) cuando hay dato.

Este tiempo aproxima el ciclo completo (revisión + producción), por lo que
es algo mayor que el "tiempo a primera decisión". Lo aproximamos como el
mejor proxy automático disponible; el dato fino se rellena en el editor.

Para no saturar CrossRef, muestreamos M artículos por revista (por defecto 15).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import (
    ARTICLES_INDEX_PATH,
    OPENALEX_MAILTO,
    REVIEW_TIMES_PATH,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CROSSREF_API = "https://api.crossref.org/works"
SAMPLE_PER_JOURNAL = 15


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _query_crossref(doi: str) -> dict | None:
    if not doi:
        return None
    # Normaliza https://doi.org/X → X
    if doi.startswith("http"):
        doi = doi.split("doi.org/", 1)[-1]
    url = f"{CROSSREF_API}/{doi}"
    headers = {"User-Agent": f"app-revistas/0.1 (mailto:{OPENALEX_MAILTO})"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("message")


def _date_from_parts(date_parts: list[list[int]] | None) -> date | None:
    if not date_parts:
        return None
    if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list):
        parts = date_parts[0]
    else:
        parts = date_parts
    try:
        y = parts[0]
        m = parts[1] if len(parts) > 1 else 1
        d = parts[2] if len(parts) > 2 else 1
        return date(y, m or 1, d or 1)
    except (ValueError, TypeError, IndexError):
        return None


def _extract_dates(msg: dict) -> tuple[date | None, date | None, date | None]:
    """Devuelve (received, accepted, published)."""
    received = _date_from_parts(((msg.get("received") or {}).get("date-parts")))
    accepted = _date_from_parts(((msg.get("accepted") or {}).get("date-parts")))
    published = _date_from_parts(((msg.get("published-online") or {}).get("date-parts")))
    if not published:
        published = _date_from_parts(((msg.get("published-print") or {}).get("date-parts")))
    if not published:
        published = _date_from_parts(((msg.get("published") or {}).get("date-parts")))
    return received, accepted, published


def build_index(sample_per_journal: int = SAMPLE_PER_JOURNAL) -> None:
    if not ARTICLES_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Falta {ARTICLES_INDEX_PATH}. Ejecuta antes: python -m pipelines.fetch_openalex"
        )

    articles = pd.read_parquet(ARTICLES_INDEX_PATH)
    # Solo los que tienen DOI
    articles = articles.dropna(subset=["doi"]).copy()

    log.info("Computando tiempos de revisión muestreando %d artículos por revista (%d total a consultar)…",
             sample_per_journal, sample_per_journal * articles["journal_id"].nunique())

    pub_weeks_by_j: dict[str, list[float]] = defaultdict(list)
    acc_weeks_by_j: dict[str, list[float]] = defaultdict(list)

    for jid, group in articles.groupby("journal_id"):
        sample = group.sample(min(sample_per_journal, len(group)), random_state=42)
        for _, row in sample.iterrows():
            try:
                msg = _query_crossref(row["doi"])
            except requests.RequestException:
                continue
            if not msg:
                continue
            received, accepted, published = _extract_dates(msg)
            if received and published:
                weeks = (published - received).days / 7.0
                if 0 < weeks < 200:  # filtra valores absurdos
                    pub_weeks_by_j[jid].append(weeks)
            if received and accepted:
                weeks_a = (accepted - received).days / 7.0
                if 0 < weeks_a < 200:
                    acc_weeks_by_j[jid].append(weeks_a)
            time.sleep(0.05)  # ~20 req/s, dentro del polite pool

    rows = []
    for jid in articles["journal_id"].unique():
        pw = pub_weeks_by_j.get(jid, [])
        aw = acc_weeks_by_j.get(jid, [])
        rows.append({
            "journal_id": jid,
            "time_to_publication_weeks_median": float(np.median(pw)) if pw else None,
            "time_to_publication_n": len(pw),
            "time_to_acceptance_weeks_median": float(np.median(aw)) if aw else None,
            "time_to_acceptance_n": len(aw),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(REVIEW_TIMES_PATH, index=False)
    n_with_pub = (df["time_to_publication_weeks_median"].notna()).sum()
    n_with_acc = (df["time_to_acceptance_weeks_median"].notna()).sum()
    log.info("✔ Guardado %s con %d revistas (%d con time-to-pub, %d con time-to-acc)",
             REVIEW_TIMES_PATH.name, len(df), n_with_pub, n_with_acc)


if __name__ == "__main__":
    build_index()
