"""Descarga metadatos de revistas y sus artículos recientes desde OpenAlex.

OpenAlex es gratuito y open-source. Para el "polite pool" (más cuota y mejor
soporte) basta con incluir `mailto=` en cada petición.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import (
    ARTICLES_INDEX_PATH,
    JOURNALS_INDEX_PATH,
    OPENALEX_ARTICLES_PER_JOURNAL,
    OPENALEX_BASE,
    OPENALEX_DIR,
    OPENALEX_MAILTO,
    TARGET_SJR_CATEGORIES,
)
from pipelines.seed_journals import SEED_JOURNALS, favorites_issn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_DIR = OPENALEX_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def _get(url: str, params: dict | None = None) -> dict:
    params = {**(params or {}), "mailto": OPENALEX_MAILTO}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _cache_get(name: str) -> dict | None:
    p = CACHE_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(name: str, data: dict) -> None:
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def resolve_source(name: str, issn_hint: str | None = None) -> dict | None:
    """Busca la 'source' (revista) en OpenAlex por ISSN o por nombre."""
    cache_key = f"source__{(issn_hint or name).replace('/', '_')}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # Por ISSN
    if issn_hint:
        try:
            data = _get(f"{OPENALEX_BASE}/sources",
                        params={"filter": f"issn:{issn_hint}", "per-page": 5})
            if data.get("results"):
                # Preferir el resultado con más works (los duplicados existen)
                best = max(data["results"], key=lambda r: r.get("works_count", 0))
                _cache_put(cache_key, best)
                return best
        except requests.RequestException as exc:
            log.warning("ISSN lookup falló para %s: %s", issn_hint, exc)

    # Por nombre
    try:
        data = _get(f"{OPENALEX_BASE}/sources",
                    params={"search": name, "per-page": 10})
        results = data.get("results", [])
        # filtrado: queremos source de tipo 'journal' y nombre parecido
        results = [r for r in results if r.get("type") == "journal"]
        if not results:
            return None
        name_lower = name.lower()
        # priorizar coincidencia exacta del display_name
        exact = [r for r in results if r.get("display_name", "").lower() == name_lower]
        best = exact[0] if exact else results[0]
        _cache_put(cache_key, best)
        return best
    except requests.RequestException as exc:
        log.warning("Name lookup falló para %s: %s", name, exc)
        return None


def fetch_works(source_id: str, n: int = OPENALEX_ARTICLES_PER_JOURNAL) -> list[dict]:
    """Trae los `n` works más recientes (artículos) de una revista."""
    cache_key = f"works__{source_id.split('/')[-1]}__n{n}"
    cached = _cache_get(cache_key)
    if cached:
        return cached.get("results", [])

    results: list[dict] = []
    per_page = 50
    cursor = "*"
    while len(results) < n:
        try:
            page = _get(f"{OPENALEX_BASE}/works", params={
                "filter": f"primary_location.source.id:{source_id.split('/')[-1]},type:article",
                "per-page": per_page,
                "cursor": cursor,
                "select": "id,doi,title,abstract_inverted_index,publication_year,authorships",
                "sort": "publication_date:desc",
            })
        except requests.RequestException as exc:
            log.warning("Fetch works falló para %s: %s", source_id, exc)
            break
        items = page.get("results", [])
        results.extend(items)
        next_cursor = page.get("meta", {}).get("next_cursor")
        if not next_cursor or len(items) < per_page:
            break
        cursor = next_cursor
        time.sleep(0.1)  # gentileza

    results = results[:n]
    _cache_put(cache_key, {"results": results})
    return results


def _abstract_from_inverted(inv: dict | None) -> str:
    """OpenAlex devuelve los abstracts como índice invertido (palabra→[posiciones])
    por restricciones de copyright. Lo reconstruimos."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, posns in inv.items():
        for p in posns:
            positions.append((p, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def build_index() -> None:
    favs = favorites_issn()
    journal_rows: list[dict] = []
    article_rows: list[dict] = []

    for name, issn_hint, note in SEED_JOURNALS:
        log.info("→ %s", name)
        src = resolve_source(name, issn_hint)
        if not src:
            log.warning("  No encontrado en OpenAlex")
            continue

        oa_id = src.get("id", "")
        journal_id = oa_id.split("/")[-1] if oa_id else name
        issns = src.get("issn") or []
        primary_issn = src.get("issn_l") or (issns[0] if issns else issn_hint)
        publisher = src.get("host_organization_name")
        is_oa = src.get("is_oa")
        apc_usd = (src.get("apc_usd") or src.get("apc_prices") or [{}])
        apc_eur = None
        if isinstance(apc_usd, list) and apc_usd:
            # algunos sources traen lista [{price, currency}]
            for entry in apc_usd:
                if isinstance(entry, dict) and entry.get("currency") in ("EUR", "USD"):
                    apc_eur = float(entry.get("price", 0)) * (0.92 if entry.get("currency") == "USD" else 1.0)
                    break
        elif isinstance(apc_usd, (int, float)):
            apc_eur = float(apc_usd) * 0.92

        summary = src.get("summary_stats", {}) or {}
        citedness = summary.get("2yr_mean_citedness")
        h_index = summary.get("h_index")

        # categorías SJR (OpenAlex incluye 'topics' / 'concepts' pero no categorías SJR;
        # solo las marcamos como 'in_target' si alguna palabra clave coincide).
        x_concepts = [c.get("display_name", "") for c in src.get("x_concepts", [])]
        in_target = any(
            any(target.lower() in c.lower() for c in x_concepts) for target in TARGET_SJR_CATEGORIES
        ) or any(
            target.lower() in name.lower() for target in [
                "neuro", "stroke", "rehab", "physi", "brain", "cogn"
            ]
        )

        journal_rows.append({
            "journal_id": journal_id,
            "name": src.get("display_name", name),
            "issn": primary_issn,
            "all_issns": ";".join(issns) if issns else "",
            "publisher": publisher,
            "openalex_2yr_mean_citedness": citedness,
            "openalex_h_index": h_index,
            "works_count": src.get("works_count"),
            "is_oa": is_oa,
            "apc_eur": apc_eur,
            "review_weeks": None,           # se rellena con dataset externo si está disponible
            "in_jcr": True,                 # placeholder; build_jcr_list.py lo corregirá
            "in_target_category": in_target,
            "is_favorite": bool(note and "favorita" in note),
            "homepage_url": src.get("homepage_url"),
            "note": note or "",
        })

        # Artículos
        works = fetch_works(oa_id, OPENALEX_ARTICLES_PER_JOURNAL)
        log.info("  %d artículos descargados", len(works))
        for w in works:
            abstract = _abstract_from_inverted(w.get("abstract_inverted_index"))
            if not (w.get("title") and abstract):
                continue
            article_rows.append({
                "article_id": w["id"].split("/")[-1],
                "journal_id": journal_id,
                "title": w.get("title", ""),
                "abstract": abstract,
                "year": w.get("publication_year"),
                "doi": w.get("doi"),
            })

    journals_df = pd.DataFrame(journal_rows)
    articles_df = pd.DataFrame(article_rows)

    JOURNALS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    journals_df.to_parquet(JOURNALS_INDEX_PATH, index=False)
    articles_df.to_parquet(ARTICLES_INDEX_PATH, index=False)

    log.info("✔ Guardado %s (%d revistas) y %s (%d artículos)",
             JOURNALS_INDEX_PATH.name, len(journals_df),
             ARTICLES_INDEX_PATH.name, len(articles_df))


if __name__ == "__main__":
    build_index()
