"""Configuración central de la app (rutas, constantes, categorías JCR objetivo)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
JCR_DIR = DATA_DIR / "jcr"
SJR_DIR = DATA_DIR / "sjr"
OPENALEX_DIR = DATA_DIR / "openalex"
MODELS_DIR = ROOT_DIR / "models"

JOURNALS_INDEX_PATH = DATA_DIR / "journals.parquet"
ARTICLES_INDEX_PATH = DATA_DIR / "articles.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings_meta.parquet"
JCR_INDEX_PATH = DATA_DIR / "jcr_index.parquet"
JCR_RANKS_PATH = DATA_DIR / "jcr_ranks.parquet"     # detalle por (revista, categoría)
SJR_INDEX_PATH = DATA_DIR / "sjr_index.parquet"
DOAJ_INDEX_PATH = DATA_DIR / "doaj_index.parquet"
SHERPA_INDEX_PATH = DATA_DIR / "sherpa_index.parquet"
REVIEW_TIMES_PATH = DATA_DIR / "review_times.parquet"
FAVORITES_PATH = DATA_DIR / "favorites.json"
OVERRIDES_PATH = DATA_DIR / "overrides.json"

# ── Categorías JCR objetivo ───────────────────────────────────────────────
TARGET_JCR_CATEGORIES: list[str] = [
    "Clinical Neurology",
    "Neurosciences",
    "Rehabilitation",
    "Radiology, Nuclear Medicine & Medical Imaging",
]

# Categorías SJR equivalentes (la nomenclatura de Scimago no es idéntica).
TARGET_SJR_CATEGORIES: list[str] = [
    "Clinical Neurology",
    "Neurology (clinical)",
    "Neuroscience (miscellaneous)",
    "Cognitive Neuroscience",
    "Behavioral Neuroscience",
    "Cellular and Molecular Neuroscience",
    "Developmental Neuroscience",
    "Sensory Systems",
    "Rehabilitation",
    "Physical Therapy, Sports Therapy and Rehabilitation",
    "Radiology, Nuclear Medicine and Imaging",
    "Neuroimaging",
]

# ── Modelo de embeddings ──────────────────────────────────────────────────
# allenai/specter funciona out-of-the-box con sentence-transformers y es
# perfecto como baseline. Para v2 podemos saltar a specter2_base + adapters.
EMBEDDING_MODEL_NAME = "allenai/specter"
EMBEDDING_DIM = 768

# ── Comportamiento del ranking ────────────────────────────────────────────
DEFAULT_TOP_K_ARTICLES = 5     # cuántos artículos similares por revista se promedian
DEFAULT_MIN_ARTICLES = 10      # revistas con menos artículos en el corpus se descartan
DEFAULT_RANKING_RESULTS = 25   # cuántas revistas devolver por defecto

# ── OpenAlex ──────────────────────────────────────────────────────────────
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAILTO = "alfre_lerin@hotmail.com"  # política de "polite pool" de OpenAlex
OPENALEX_ARTICLES_PER_JOURNAL = 200  # en local sube a 200; en el repo viaja un mini-índice con 50/revista

# ── Excel: orden de columnas para exportar manteniendo formato del usuario ─
EXCEL_EXPORT_COLUMNS: list[str] = [
    "Preferencia de envío",
    "Nombre de revista",
    "Cuartil",
    "IF",
    "Coste de publicación",
    "Editorial",
    "Semanas max de promedio peer review",
    "Rango de coincidencia con la temática",
]


@dataclass(frozen=True)
class FilterDefaults:
    """Valores por defecto del panel de filtros."""
    min_quartile: str = "Q2"      # Q1, Q2, Q3, Q4, "Cualquiera"
    min_if: float = 0.0
    max_apc_eur: float | None = None
    only_open_access: bool = False
    max_review_weeks: int | None = None
    excluded_publishers: list[str] = field(default_factory=list)


FILTER_DEFAULTS = FilterDefaults()
