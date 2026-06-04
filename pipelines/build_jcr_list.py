"""Lee los CSV/XLSX de JCR que el usuario haya colocado en data/jcr/
(propietario de Clarivate, no se sube al repo) y construye dos índices:

  - data/jcr_index.parquet  → una fila por ISSN, con el MEJOR cuartil y el
                              JIF de la categoría top (para backward-compat
                              con filtros simples).
  - data/jcr_ranks.parquet  → una fila por (ISSN, categoría), con la
                              posición exacta en cada categoría (ej. 2/210).
                              Esto permite mostrar 'Q1 (2/210) Clinical
                              Neurology · Q1 (8/197) Neurosciences'.

Las columnas que JCR exporta varían año a año, así que el parser es flexible.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from core.config import JCR_DIR, JCR_INDEX_PATH, JCR_RANKS_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


NAME_COLS = ["Full Journal Title", "Journal name", "Journal Title", "Title", "Journal Name"]
ISSN_COLS = ["ISSN", "Print ISSN", "Journal ISSN"]
EISSN_COLS = ["eISSN", "Electronic ISSN", "EISSN"]
JIF_COLS = [
    "2024 JIF", "2023 JIF", "2022 JIF", "2021 JIF",
    "JIF", "Journal Impact Factor", "Impact Factor",
]
QUART_COLS = ["JIF Quartile", "Quartile", "JCR Quartile", "Category Quartile"]
CAT_COLS = ["Category", "Categories", "Category Name", "Category Description"]
RANK_COLS = ["Rank", "JIF Rank", "Rank in Category", "Category Rank"]
TOTAL_COLS = ["Journals in Category", "Total Journals", "Category Size", "Total in Category"]
PERCENTILE_COLS = ["JIF Percentile", "Percentile", "JIF Percentile in Category"]
JCI_COLS = ["2024 JCI", "2023 JCI", "JCI", "Journal Citation Indicator"]


def _pick(row: dict, options: list[str]) -> str | None:
    for o in options:
        if o in row and pd.notna(row[o]) and str(row[o]).strip():
            return str(row[o]).strip()
    return None


def _norm_issn(v: str | None) -> str | None:
    if not v:
        return None
    digits = re.sub(r"\D", "", v)
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    return None


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace(",", ".").replace("<", "").replace(">", "")
    if not s or s in ("—", "-", "N/A", "n/a", "Not Available"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _to_int(v) -> int | None:
    f = _to_float(v)
    if f is None:
        return None
    return int(round(f))


def _parse_rank(rank_raw: str | None, total_raw: str | None) -> tuple[int | None, int | None]:
    """Acepta '12', '12/231', '12 of 231'."""
    if not rank_raw:
        return None, _to_int(total_raw)
    s = str(rank_raw).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return _to_int(a), _to_int(b)
    if " of " in s:
        a, b = s.split(" of ", 1)
        return _to_int(a), _to_int(b)
    return _to_int(s), _to_int(total_raw)


def read_one(path: Path) -> pd.DataFrame:
    """Lee un CSV/XLSX de JCR siendo tolerante a:
       - línea(s) iniciales de metadatos del export
       - coma sobrante al final de cada fila (típica del export JCR 2024+)
       - encoding y BOM variables
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for skip in (0, 1, 2, 3):
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    skiprows=skip,
                    on_bad_lines="skip",
                    index_col=False,   # ← evita que el coma sobrante desplace las columnas
                )
                if any(c in df.columns for c in (NAME_COLS + ISSN_COLS)):
                    # Limpia columnas residuales tipo "Unnamed: 12" creadas por la coma sobrante
                    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"No se pudo parsear {path}")


def build_index() -> None:
    JCR_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in JCR_DIR.iterdir()
             if p.suffix.lower() in (".csv", ".xlsx", ".xls") and not p.name.startswith(".")]
    if not files:
        log.info("No hay CSV/XLSX de JCR en %s. La app correrá usando solo SJR/OpenAlex.", JCR_DIR)
        return

    # Lista intermedia por (revista, categoría) ANTES de explotar a ISSNs.
    # Necesitamos esto para calcular el rank dentro de cada categoría
    # cuando JCR no lo exporta directamente.
    cat_rows: list[dict] = []
    for path in files:
        log.info("Leyendo %s", path.name)
        try:
            df = read_one(path)
        except ValueError as exc:
            log.warning("  Saltado: %s", exc)
            continue

        log.info("  Columnas detectadas: %s", ", ".join(df.columns[:8]))
        n_in_cat = 0
        for _, r in df.iterrows():
            row = r.to_dict()
            name = _pick(row, NAME_COLS)
            if not name:
                continue
            issn = _norm_issn(_pick(row, ISSN_COLS))
            eissn = _norm_issn(_pick(row, EISSN_COLS))
            jif = _to_float(_pick(row, JIF_COLS))
            jci = _to_float(_pick(row, JCI_COLS))
            q = _pick(row, QUART_COLS)
            q = q if q and q.startswith("Q") else None
            cat = _pick(row, CAT_COLS) or path.stem.upper()
            rank_raw = _pick(row, RANK_COLS)
            total_raw = _pick(row, TOTAL_COLS)
            rank, total = _parse_rank(rank_raw, total_raw)
            pct = _to_float(_pick(row, PERCENTILE_COLS))

            cat_rows.append({
                "name": name,
                "issn": issn,
                "eissn": eissn,
                "jcr_category": cat,
                "jcr_jif": jif,
                "jcr_jci": jci,
                "jcr_quartile": q,
                "jcr_rank": rank,
                "jcr_category_size": total,
                "jcr_percentile": pct,
            })
            n_in_cat += 1
        log.info("  → %d revistas en esta categoría", n_in_cat)

    if not cat_rows:
        log.warning("Ningún registro válido encontrado en data/jcr/")
        return

    cat_df = pd.DataFrame(cat_rows)

    # Si no nos dieron rank, lo calculamos ordenando por JIF dentro de cada categoría
    log.info("Calculando posición (rank) dentro de cada categoría…")
    cat_df["jcr_jif"] = cat_df["jcr_jif"].astype(float)

    def _fill_rank(group: pd.DataFrame) -> pd.DataFrame:
        if group["jcr_rank"].isna().all():
            sorted_g = group.sort_values("jcr_jif", ascending=False, na_position="last")
            sorted_g["jcr_rank"] = range(1, len(sorted_g) + 1)
            return sorted_g
        return group

    cat_df = cat_df.groupby("jcr_category", group_keys=False).apply(_fill_rank)
    # Tamaño de categoría = número de revistas en ese CSV
    cat_size = cat_df.groupby("jcr_category").size().to_dict()
    cat_df["jcr_category_size"] = cat_df.apply(
        lambda r: r["jcr_category_size"] if pd.notna(r["jcr_category_size"]) else cat_size.get(r["jcr_category"]),
        axis=1,
    )

    # Ahora explotamos a (issn, eissn) → una fila por cada ISSN registrado
    rank_rows: list[dict] = []
    for _, r in cat_df.iterrows():
        for ident in (r["issn"], r["eissn"]):
            if ident:
                rank_rows.append({
                    "issn": ident,
                    "name": r["name"],
                    "jcr_category": r["jcr_category"],
                    "jcr_jif": r["jcr_jif"],
                    "jcr_jci": r["jcr_jci"],
                    "jcr_quartile": r["jcr_quartile"],
                    "jcr_rank": int(r["jcr_rank"]) if pd.notna(r["jcr_rank"]) else None,
                    "jcr_category_size": int(r["jcr_category_size"]) if pd.notna(r["jcr_category_size"]) else None,
                    "jcr_percentile": r["jcr_percentile"],
                })

    ranks_df = pd.DataFrame(rank_rows).drop_duplicates(
        subset=["issn", "jcr_category"]
    ).reset_index(drop=True)

    JCR_RANKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranks_df.to_parquet(JCR_RANKS_PATH, index=False)
    log.info("✔ %s con %d filas (revista × categoría)", JCR_RANKS_PATH.name, len(ranks_df))

    # Versión agregada (best quartile, max JIF) para los filtros sencillos
    def _best_q(s: pd.Series) -> str | None:
        opts = [x for x in s if isinstance(x, str) and x.startswith("Q")]
        return min(opts) if opts else None

    agg = ranks_df.groupby("issn", dropna=False).agg(
        name=("name", "first"),
        jcr_jif=("jcr_jif", "max"),
        jcr_quartile=("jcr_quartile", _best_q),
        jcr_category=("jcr_category", lambda s: "; ".join(sorted({x for x in s if isinstance(x, str)}))),
        jcr_best_rank=("jcr_rank", "min"),
        jcr_best_percentile=("jcr_percentile", "max"),
    ).reset_index()

    agg.to_parquet(JCR_INDEX_PATH, index=False)
    log.info("✔ %s con %d revistas únicas (datos JCR locales, no se suben al repo)",
             JCR_INDEX_PATH.name, len(agg))


if __name__ == "__main__":
    build_index()
