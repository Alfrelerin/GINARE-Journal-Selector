"""Descarga el ranking SJR (Scimago Journal Rank) en CSV y construye un
índice con cuartil + score por revista.

Scimago publica anualmente CSV gratuitos por categoría en:
   https://www.scimagojr.com/journalrank.php?category=NNNN&out=xls

Como el ID de categoría no es estable y Scimago a veces cambia la URL,
el approach robusto es: si el usuario tiene CSVs en `data/sjr/`, los
parseamos. Si no, intentamos descargar la última disponible.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests

from core.config import SJR_DIR, SJR_INDEX_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# URLs canónicas por categoría (Scimago). Los IDs son estables en la práctica.
SJR_CATEGORY_URLS = {
    "Clinical Neurology": "https://www.scimagojr.com/journalrank.php?category=2728&out=xls",
    "Neuroscience": "https://www.scimagojr.com/journalrank.php?category=2800&out=xls",
    "Rehabilitation": "https://www.scimagojr.com/journalrank.php?category=3612&out=xls",
    "Radiology and Imaging": "https://www.scimagojr.com/journalrank.php?category=2741&out=xls",
}


def download_category(label: str, url: str) -> Path | None:
    """Descarga el CSV de una categoría SJR y lo guarda en data/sjr/.

    Scimago devuelve un fichero .xls pero en realidad es un CSV con
    separador `;`. Lo guardamos como .csv para evitar confusión.
    """
    safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    target = SJR_DIR / f"sjr_{safe}.csv"
    if target.exists():
        log.info("[%s] ya existe en %s — skip descarga", label, target)
        return target
    try:
        log.info("Descargando SJR [%s] …", label)
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        target.write_bytes(r.content)
        log.info("  Guardado en %s", target)
        return target
    except requests.RequestException as exc:
        log.warning("Fallo descargando %s: %s", label, exc)
        return None


def parse_sjr_csv(path: Path, category_label: str) -> pd.DataFrame:
    """Parsea el CSV de Scimago. Columnas típicas:
       Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;...
    """
    # Probar separador ; primero
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_csv(path, sep=";", encoding="latin-1")

    df = df.rename(columns={
        "Title": "name",
        "Issn": "issns_raw",
        "SJR": "sjr_score",
        "SJR Best Quartile": "sjr_quartile",
        "H index": "sjr_h_index",
    })

    rows = []
    for _, r in df.iterrows():
        raw = str(r.get("issns_raw") or "")
        # Scimago concatena ISSN y eISSN como "01234567, 76543210" o "01234567"
        issns = [s.strip() for s in re.split(r"[,;\s]+", raw) if s.strip()]
        # Formatea ISSN como NNNN-NNNN
        norm: list[str] = []
        for i in issns:
            digits = re.sub(r"\D", "", i)
            if len(digits) == 8:
                norm.append(f"{digits[:4]}-{digits[4:]}")
        for issn in norm or [None]:
            rows.append({
                "issn": issn,
                "name": r.get("name"),
                "sjr_score": _to_float(r.get("sjr_score")),
                "sjr_quartile": r.get("sjr_quartile") if str(r.get("sjr_quartile") or "").startswith("Q") else None,
                "sjr_h_index": _to_int(r.get("sjr_h_index")),
                "sjr_categories": category_label,
            })
    return pd.DataFrame(rows)


def _to_float(v) -> float | None:
    try:
        if pd.isna(v):
            return None
        return float(str(v).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _to_int(v) -> int | None:
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def build_index() -> None:
    SJR_DIR.mkdir(parents=True, exist_ok=True)

    # Si hay CSVs colocados manualmente, los respetamos. Si no, descargamos.
    local_csvs = sorted(SJR_DIR.glob("sjr_*.csv"))
    if not local_csvs:
        for label, url in SJR_CATEGORY_URLS.items():
            download_category(label, url)
        local_csvs = sorted(SJR_DIR.glob("sjr_*.csv"))

    if not local_csvs:
        log.warning("No hay CSVs de SJR en %s. La app correrá sin cuartil SJR.", SJR_DIR)
        return

    frames = []
    for p in local_csvs:
        label = p.stem.replace("sjr_", "").replace("_", " ").title()
        log.info("Parseando %s (%s)", p.name, label)
        df = parse_sjr_csv(p, label)
        frames.append(df)

    if not frames:
        return

    all_df = pd.concat(frames, ignore_index=True)
    # Si una revista aparece en varias categorías, nos quedamos con el cuartil
    # mejor y agregamos las categorías
    def _best_quartile(s: pd.Series) -> str | None:
        opts = [x for x in s if isinstance(x, str) and x.startswith("Q")]
        return min(opts) if opts else None

    agg = all_df.groupby("issn", dropna=False).agg(
        name=("name", "first"),
        sjr_score=("sjr_score", "max"),
        sjr_quartile=("sjr_quartile", _best_quartile),
        sjr_h_index=("sjr_h_index", "max"),
        sjr_categories=("sjr_categories", lambda s: "; ".join(sorted(set(s)))),
    ).reset_index()

    agg = agg.dropna(subset=["issn"])
    agg.to_parquet(SJR_INDEX_PATH, index=False)
    log.info("✔ Guardado %s con %d entradas", SJR_INDEX_PATH.name, len(agg))


if __name__ == "__main__":
    build_index()
