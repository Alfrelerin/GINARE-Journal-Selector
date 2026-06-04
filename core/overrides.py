"""Sistema de verificaciones manuales por revista.

Cada revista (ISSN) puede tener un conjunto de campos verificados por el
usuario. Los campos verificados toman precedencia sobre cualquier dato
automático (DOAJ, OpenAlex…).

Persistencia: `data/overrides.json` se sube al repo público para que los
beneficios sean compartidos. La estructura es estable y tolerante a campos
adicionales.

Ejemplo de fichero:
{
  "0006-8950": {
    "oa_model": "subscription",
    "apc_eur": 3990,
    "time_to_first_decision_weeks": 8,
    "acceptance_rate_pct": 12,
    "notes": "Editor responde rápido en el primer round.",
    "verified_at": "2026-05-29",
    "verified_by": "alfre_lerin",
    "verified_fields": ["oa_model", "apc_eur", "time_to_first_decision_weeks"]
  }
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.config import DATA_DIR

OVERRIDES_PATH = DATA_DIR / "overrides.json"

# Campos que pueden ser editados/verificados manualmente
EDITABLE_FIELDS: tuple[str, ...] = (
    "oa_model",                       # diamond | gold | hybrid | subscription
    "apc_eur",                        # float | None
    "time_to_first_decision_weeks",   # int | None
    "acceptance_rate_pct",            # float (0..100) | None
    "homepage_url",                   # str | None — para corregir URLs erróneas
    "notes",                          # str | None
    "publisher",                      # str | None — editorial
    "impact_factor",                  # float | None — IF manual (gana sobre JCR/OpenAlex)
    "quartile",                       # "Q1".."Q4" | None — cuartil manual (gana sobre JCR)
    "manual_jcr_category",            # str | None — categoría/área JCR (override manual)
    "manual_jcr_rank",                # int | None — posición en la categoría
    "manual_jcr_total",               # int | None — nº total de revistas en la categoría
)

OA_MODELS = ("diamond", "gold", "hybrid", "subscription", "unknown")


@dataclass
class JournalOverride:
    issn: str
    oa_model: str | None = None
    apc_eur: float | None = None
    time_to_first_decision_weeks: int | None = None
    acceptance_rate_pct: float | None = None
    homepage_url: str | None = None
    notes: str | None = None
    publisher: str | None = None
    impact_factor: float | None = None
    quartile: str | None = None
    manual_jcr_category: str | None = None
    manual_jcr_rank: int | None = None
    manual_jcr_total: int | None = None
    verified_at: str | None = None
    verified_by: str | None = None
    verified_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, issn: str, d: dict) -> "JournalOverride":
        return cls(
            issn=issn,
            oa_model=d.get("oa_model"),
            apc_eur=d.get("apc_eur"),
            time_to_first_decision_weeks=d.get("time_to_first_decision_weeks"),
            acceptance_rate_pct=d.get("acceptance_rate_pct"),
            homepage_url=d.get("homepage_url"),
            notes=d.get("notes"),
            publisher=d.get("publisher"),
            impact_factor=d.get("impact_factor"),
            quartile=d.get("quartile"),
            manual_jcr_category=d.get("manual_jcr_category"),
            manual_jcr_rank=d.get("manual_jcr_rank"),
            manual_jcr_total=d.get("manual_jcr_total"),
            verified_at=d.get("verified_at"),
            verified_by=d.get("verified_by"),
            verified_fields=list(d.get("verified_fields", [])),
        )

    def to_dict(self) -> dict:
        return {
            "oa_model": self.oa_model,
            "apc_eur": self.apc_eur,
            "time_to_first_decision_weeks": self.time_to_first_decision_weeks,
            "acceptance_rate_pct": self.acceptance_rate_pct,
            "homepage_url": self.homepage_url,
            "notes": self.notes,
            "publisher": self.publisher,
            "impact_factor": self.impact_factor,
            "quartile": self.quartile,
            "manual_jcr_category": self.manual_jcr_category,
            "manual_jcr_rank": self.manual_jcr_rank,
            "manual_jcr_total": self.manual_jcr_total,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "verified_fields": list(self.verified_fields),
        }

    def is_field_verified(self, field_name: str) -> bool:
        return field_name in self.verified_fields


def load_overrides() -> dict[str, JournalOverride]:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {issn: JournalOverride.from_dict(issn, d) for issn, d in data.items()}


def save_overrides(overrides: dict[str, JournalOverride]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    serial = {issn: ov.to_dict() for issn, ov in sorted(overrides.items())}
    OVERRIDES_PATH.write_text(
        json.dumps(serial, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_override(
    issn: str,
    fields: dict,
    verified_fields: Iterable[str],
    verified_by: str = "user",
) -> JournalOverride:
    """Actualiza o crea el override para una revista, marcando los campos
    pasados como verificados con la fecha de hoy."""
    overrides = load_overrides()
    existing = overrides.get(issn) or JournalOverride(issn=issn)

    today = date.today().isoformat()
    new_verified = set(existing.verified_fields) | set(verified_fields)

    for k, v in fields.items():
        if k in EDITABLE_FIELDS:
            setattr(existing, k, v)

    existing.verified_at = today
    existing.verified_by = verified_by
    existing.verified_fields = sorted(new_verified)

    overrides[issn] = existing
    save_overrides(overrides)
    return existing


def merge_overrides_into_journals(journals: pd.DataFrame) -> pd.DataFrame:
    """Sobreescribe los datos de un DataFrame de revistas con los valores
    verificados manualmente. Añade columnas booleanas `verified_<campo>`
    para que la UI pueda mostrar el badge ✅."""
    if journals.empty:
        return journals

    overrides = load_overrides()
    if not overrides:
        return journals

    df = journals.copy()

    # Aseguramos columnas que pueden no existir aún
    for col in EDITABLE_FIELDS:
        if col not in df.columns:
            df[col] = None
    df["verified_at"] = None
    df["verified_by"] = None
    for col in EDITABLE_FIELDS:
        df[f"verified_{col}"] = False

    for idx, row in df.iterrows():
        issn = row.get("issn")
        if not issn or issn not in overrides:
            continue
        ov = overrides[issn]
        for k in EDITABLE_FIELDS:
            val = getattr(ov, k)
            if val is not None and val != "":
                df.at[idx, k] = val
                if ov.is_field_verified(k):
                    df.at[idx, f"verified_{k}"] = True
        df.at[idx, "verified_at"] = ov.verified_at
        df.at[idx, "verified_by"] = ov.verified_by

    return df
