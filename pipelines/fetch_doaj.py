"""Consulta la API pública de DOAJ por cada ISSN del índice y construye
un índice autoritativo de modelo OA + APC oficial.

DOAJ (Directory of Open Access Journals) es la **fuente canónica** para
saber si una revista es realmente Open Access. Solo lista revistas que
han pasado su control de calidad (no predatorias, peer review real…).

API: https://doaj.org/api/v2/search/journals/issn:NNNN-NNNN
Sin autenticación. Rate limit suave (1-2 req/s recomendado).

Clasificación del oa_model:
  - diamond:      DOAJ-listada y APC declarado = 0
  - gold:         DOAJ-listada y APC > 0
  - hybrid:       no en DOAJ pero el publisher es uno de los grandes
                  (Elsevier, Springer, Wiley, Taylor & Francis, SAGE, Oxford…)
  - subscription: no en DOAJ y publisher no ofrece OA híbrido claramente
  - unknown:      no se puede determinar
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import DATA_DIR, JOURNALS_INDEX_PATH, OPENALEX_MAILTO

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DOAJ_INDEX_PATH = DATA_DIR / "doaj_index.parquet"
DOAJ_CACHE = DATA_DIR / "doaj" / "cache"
DOAJ_CACHE.mkdir(parents=True, exist_ok=True)

DOAJ_API = "https://doaj.org/api/v2/search/journals"

# Publishers conocidos por ofrecer modelo híbrido (subscripción + OA pagado).
# Lista no exhaustiva, suficiente para mejorar la heurística.
HYBRID_PUBLISHERS = {
    "elsevier", "elsevier bv",
    "springer", "springer nature", "nature portfolio", "springer-verlag",
    "wiley", "wiley-blackwell", "john wiley & sons",
    "taylor & francis", "taylor and francis", "routledge", "informa",
    "sage", "sage publishing", "sage publications",
    "oxford university press", "oup",
    "cambridge university press", "cup",
    "lippincott williams & wilkins", "wolters kluwer",
    "ios press", "cell press",
    "ama", "american medical association",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _query_doaj(issn: str) -> dict | None:
    cache_file = DOAJ_CACHE / f"{issn}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)

    url = f"{DOAJ_API}/issn:{issn}"
    resp = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": f"app-revistas/0.1 (mailto:{OPENALEX_MAILTO})"},
    )
    if resp.status_code == 404:
        cache_file.write_text(json.dumps({"results": []}), encoding="utf-8")
        return {"results": []}
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _extract_doaj_fields(result: dict) -> dict:
    """Aplana la respuesta de DOAJ a un dict plano con los campos que
    nos interesan."""
    bibjson = result.get("bibjson", {}) or {}
    apc = bibjson.get("apc", {}) or {}
    max_apc = apc.get("max") or []
    apc_currency = None
    apc_amount = None
    if isinstance(max_apc, list) and max_apc:
        apc_amount = max_apc[0].get("price")
        apc_currency = max_apc[0].get("currency")

    has_apc = apc.get("has_apc")
    # Conversión simple a EUR; los detalles los maneja el override manual
    apc_eur = None
    if apc_amount is not None:
        if apc_currency == "EUR":
            apc_eur = float(apc_amount)
        elif apc_currency == "USD":
            apc_eur = float(apc_amount) * 0.92
        elif apc_currency == "GBP":
            apc_eur = float(apc_amount) * 1.17
        else:
            apc_eur = float(apc_amount)  # mejor algo que nada

    license_info = bibjson.get("license", []) or []
    license_types = [l.get("type") for l in license_info if l.get("type")]

    plagiarism = bibjson.get("plagiarism", {}) or {}

    return {
        "doaj_indexed": True,
        "doaj_has_apc": has_apc,
        "doaj_apc_eur": apc_eur,
        "doaj_apc_amount": apc_amount,
        "doaj_apc_currency": apc_currency,
        "doaj_licenses": ";".join(license_types),
        "doaj_plagiarism_detection": plagiarism.get("detection"),
        "doaj_oa_start_year": bibjson.get("oa_start"),
        "doaj_publisher": (bibjson.get("publisher", {}) or {}).get("name"),
    }


def _classify_oa_model(doaj_row: dict | None, publisher: str | None) -> str:
    """Aplica la heurística para clasificar el modelo OA."""
    if doaj_row is not None and doaj_row.get("doaj_indexed"):
        if doaj_row.get("doaj_has_apc") is False or (
            doaj_row.get("doaj_apc_eur") in (0, None)
            and doaj_row.get("doaj_has_apc") is False
        ):
            return "diamond"
        if doaj_row.get("doaj_apc_eur") and doaj_row["doaj_apc_eur"] > 0:
            return "gold"
        # DOAJ-listada pero sin APC declarado → asumimos gold con APC desconocido
        return "gold"

    # No DOAJ-listada: vemos si el publisher es de los que ofrecen híbrido
    pub_norm = (publisher or "").lower().strip()
    if any(hp in pub_norm for hp in HYBRID_PUBLISHERS):
        return "hybrid"
    return "subscription"


def build_index() -> None:
    if not JOURNALS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Falta {JOURNALS_INDEX_PATH}. Ejecuta primero: python -m pipelines.fetch_openalex"
        )

    journals = pd.read_parquet(JOURNALS_INDEX_PATH)
    log.info("Consultando DOAJ para %d revistas…", len(journals))

    rows = []
    for idx, j in journals.iterrows():
        issn = j.get("issn")
        publisher = j.get("publisher")
        if not issn:
            rows.append({"issn": None, "oa_model": "unknown"})
            continue
        try:
            data = _query_doaj(issn)
        except requests.RequestException as exc:
            log.warning("DOAJ falló para %s: %s", issn, exc)
            rows.append({
                "issn": issn,
                "doaj_indexed": False,
                "oa_model": _classify_oa_model(None, publisher),
            })
            continue

        results = (data or {}).get("results", [])
        if not results:
            # No está en DOAJ → puede ser subscription o hybrid
            rows.append({
                "issn": issn,
                "doaj_indexed": False,
                "oa_model": _classify_oa_model(None, publisher),
            })
        else:
            fields = _extract_doaj_fields(results[0])
            fields["issn"] = issn
            fields["oa_model"] = _classify_oa_model(fields, publisher)
            rows.append(fields)
            log.info("  ✓ %-40s → %s (APC %s)",
                     j.get("name", "")[:40],
                     fields["oa_model"],
                     f"{fields.get('doaj_apc_eur'):.0f}€" if fields.get("doaj_apc_eur") else "—")

        time.sleep(0.4)  # rate-limit suave

    df = pd.DataFrame(rows)
    # Garantiza columnas siempre presentes
    for col in ("doaj_indexed", "doaj_apc_eur", "doaj_has_apc", "oa_model"):
        if col not in df.columns:
            df[col] = None
    df = df.dropna(subset=["issn"])
    df.to_parquet(DOAJ_INDEX_PATH, index=False)
    log.info("✔ Guardado %s con %d entradas", DOAJ_INDEX_PATH.name, len(df))


if __name__ == "__main__":
    build_index()
