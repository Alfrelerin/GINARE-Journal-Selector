"""Exporta el ranking a un Excel que mantiene el formato del fichero de
referencia del usuario."""
from __future__ import annotations

import io
from typing import Iterable

import pandas as pd

from core.config import EXCEL_EXPORT_COLUMNS
from core.recommender import JournalRecommendation


def _coincidencia_label(score: float) -> str:
    """Replica la columna 'Rango de coincidencia con la temática' del Excel
    original mapeando la similitud semántica a etiquetas en castellano."""
    if score >= 0.78:
        return "Muy alto"
    if score >= 0.68:
        return "Alto"
    if score >= 0.58:
        return "Medio"
    if score >= 0.48:
        return "Bajo"
    return "Muy bajo"


def recommendations_to_dataframe(recs: Iterable[JournalRecommendation]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(recs, start=1):
        rows.append({
            "Preferencia de envío": i,
            "Nombre de revista": r.name,
            "Cuartil": r.quartile or "—",
            "IF": r.impact_factor if r.impact_factor is not None else "—",
            "Coste de publicación": (
                "No fees/Open Access" if r.apc_eur is None or r.apc_eur == 0
                else round(r.apc_eur, 2)
            ),
            "Editorial": (r.publisher or "—").upper(),
            "Semanas max de promedio peer review": (
                round(r.time_to_first_decision_weeks)
                if r.time_to_first_decision_weeks else "—"
            ),
            "Rango de coincidencia con la temática": _coincidencia_label(r.topic_similarity),
        })
    df = pd.DataFrame(rows, columns=EXCEL_EXPORT_COLUMNS)
    return df


def to_excel_bytes(recs: Iterable[JournalRecommendation]) -> bytes:
    df = recommendations_to_dataframe(recs)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Recomendaciones")
        ws = writer.sheets["Recomendaciones"]
        # Ancho de columnas razonable
        widths = [20, 45, 10, 10, 22, 22, 24, 32]
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    buf.seek(0)
    return buf.getvalue()
