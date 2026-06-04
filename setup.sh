#!/usr/bin/env bash
# Setup multiplataforma para Mac/Linux. Usa uv (super rápido).
# Uso:  bash setup.sh   (o ./setup.sh tras chmod +x setup.sh)
set -euo pipefail

cd "$(dirname "$0")"

echo "🔧 Setup del recomendador de revistas"
echo "======================================="

# 1) Instalar uv si no está
if ! command -v uv &> /dev/null; then
    echo "📦 Instalando uv (gestor de Python ultra-rápido)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Cargar el PATH actualizado en esta misma shell
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Crear el entorno virtual con Python 3.12 (compatible con torch + SPECTER)
echo "🐍 Creando entorno virtual (.venv) con Python 3.12…"
uv venv --python 3.12 .venv

# 3) Activar e instalar dependencias
echo "📚 Instalando dependencias (torch, streamlit, sentence-transformers…)…"
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "✅ Setup completado."
echo ""
echo "──────────────────────────────────────────────────────────"
echo "Siguientes pasos:"
echo ""
echo "  1) Activa el entorno virtual cada vez que abras terminal:"
echo "       source .venv/bin/activate"
echo ""
echo "  2) (Opcional, recomendado) Construye el índice completo:"
echo "       python -m pipelines.run_all"
echo "     · Descarga metadatos de OpenAlex"
echo "     · Lee tus CSV de JCR de data/jcr/ (si los pones)"
echo "     · Calcula embeddings SPECTER (10-90 min, primera vez)"
echo ""
echo "  3) Lanza la app:"
echo "       streamlit run app.py"
echo "     Se abrirá en http://localhost:8501"
echo "──────────────────────────────────────────────────────────"
