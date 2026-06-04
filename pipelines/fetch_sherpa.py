"""Consulta Sherpa Romeo (Jisc) por cada ISSN para obtener la política de
auto-archivo y opciones OA.

Requiere una API key gratuita: https://v2.sherpa.ac.uk/cgi/register

Configuración:
  - Crea un fichero `.env` en la raíz del proyecto con:
        SHERPA_API_KEY=tu_key_aqui
  - O exporta la variable antes de correr el script:
        export SHERPA_API_KEY=...

Si no tienes key, el pipeline se salta sin error y la app sigue funcionando
sin la columna de auto-archivo.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import DATA_DIR, JOURNALS_INDEX_PATH, SHERPA_INDEX_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SHERPA_API = "https://v2.sherpa.ac.uk/cgi/retrieve_by_id"
SHERPA_CACHE = DATA_DIR / "sherpa" / "cache"
SHERPA_CACHE.mkdir(parents=True, exist_ok=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _query_sherpa(issn: str, api_key: str) -> dict | None:
    cache_file = SHERPA_CACHE / f"{issn}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)

    params = {
        "item-type": "publication",
        "id-type": "issn",
        "id": issn,
        "format": "Json",
        "api-key": api_key,
    }
    resp = requests.get(SHERPA_API, params=params, timeout=20)
    if resp.status_code == 404:
        cache_file.write_text(json.dumps({"items": []}), encoding="utf-8")
        return {"items": []}
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _summarise_policies(item: dict) -> dict:
    """Resume las políticas que Sherpa devuelve. Lo más útil para un autor:
       - puede subir preprint? postprint? versión publicada?
       - hay embargo?
       - color de la revista (Sherpa usa 'open', 'unrestricted', etc.)
    """
    policies = item.get("publisher_policy", []) or []
    out = {
        "sherpa_can_archive_preprint": False,
        "sherpa_can_archive_postprint": False,
        "sherpa_can_archive_published": False,
        "sherpa_embargo_months": None,
        "sherpa_summary": "",
    }
    summaries = []
    for p in policies:
        for permitted in p.get("permitted_oa", []) or []:
            article_version = permitted.get("article_version", [])
            if "submitted" in article_version:
                out["sherpa_can_archive_preprint"] = True
            if "accepted" in article_version:
                out["sherpa_can_archive_postprint"] = True
            if "published" in article_version:
                out["sherpa_can_archive_published"] = True
            for emb in permitted.get("embargo", []) or []:
                amt = emb.get("amount")
                units = emb.get("units")
                if amt and units == "months":
                    out["sherpa_embargo_months"] = min(
                        out["sherpa_embargo_months"] or amt, amt
                    )
                elif amt and units == "years":
                    out["sherpa_embargo_months"] = min(
                        out["sherpa_embargo_months"] or amt * 12, amt * 12
                    )
        if p.get("internal_moniker"):
            summaries.append(p["internal_moniker"])
    out["sherpa_summary"] = "; ".join(summaries[:3])
    return out


def build_index() -> None:
    load_dotenv()
    api_key = os.environ.get("SHERPA_API_KEY", "").strip()
    if not api_key:
        log.warning(
            "Sin SHERPA_API_KEY. Saltando consulta a Sherpa Romeo. "
            "Regístrate en https://v2.sherpa.ac.uk/cgi/register para obtener una key gratuita."
        )
        # Aún así, guardamos un parquet vacío para que data_loader no falle
        pd.DataFrame(columns=["issn"]).to_parquet(SHERPA_INDEX_PATH, index=False)
        return

    if not JOURNALS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Falta {JOURNALS_INDEX_PATH}. Ejecuta primero: python -m pipelines.fetch_openalex"
        )

    journals = pd.read_parquet(JOURNALS_INDEX_PATH)
    log.info("Consultando Sherpa Romeo para %d revistas…", len(journals))

    rows = []
    for _, j in journals.iterrows():
        issn = j.get("issn")
        if not issn:
            continue
        try:
            data = _query_sherpa(issn, api_key)
        except requests.RequestException as exc:
            log.warning("Sherpa falló para %s: %s", issn, exc)
            continue

        items = (data or {}).get("items", [])
        if not items:
            rows.append({"issn": issn, "sherpa_indexed": False})
        else:
            summary = _summarise_policies(items[0])
            summary["issn"] = issn
            summary["sherpa_indexed"] = True
            rows.append(summary)
            log.info("  ✓ %-40s → preprint=%s postprint=%s",
                     j.get("name", "")[:40],
                     summary["sherpa_can_archive_preprint"],
                     summary["sherpa_can_archive_postprint"])

        time.sleep(0.4)

    df = pd.DataFrame(rows)
    df.to_parquet(SHERPA_INDEX_PATH, index=False)
    log.info("✔ Guardado %s con %d entradas", SHERPA_INDEX_PATH.name, len(df))


if __name__ == "__main__":
    build_index()
