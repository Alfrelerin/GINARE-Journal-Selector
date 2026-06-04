"""Diagnóstico de carga de SPECTER fuera de Streamlit.

Uso (con el .venv activado):
    python diagnose_specter.py

Imprime cada paso con su tiempo. Si algo se queda colgado, a los 30 s
vuelca automáticamente la traza mostrando la línea exacta donde está parado.
Para si tarda demasiado con Ctrl+C (también imprime dónde estaba).
"""
import os
import sys
import time
import faulthandler

# Si a los 30 s seguimos vivos, volcar la pila de todos los hilos a stderr.
faulthandler.dump_traceback_later(30, repeat=True)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def step(msg):
    print(f"\n>>> {msg}", flush=True)
    return time.time()


t = step("1) Importando torch…")
import torch
print(f"    torch {torch.__version__}  ({time.time()-t:.1f}s)", flush=True)
print(f"    MPS disponible: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}", flush=True)

t = step("2) Importando sentence_transformers / transformers…")
import sentence_transformers
import transformers
from sentence_transformers import SentenceTransformer
print(f"    sentence-transformers {sentence_transformers.__version__}, "
      f"transformers {transformers.__version__}  ({time.time()-t:.1f}s)", flush=True)

from huggingface_hub import constants
cache = constants.HF_HUB_CACHE
print(f"    Caché HF: {cache}", flush=True)
specter_dir = os.path.join(cache, "models--allenai--specter")
print(f"    ¿specter en caché?: {os.path.isdir(specter_dir)}  -> {specter_dir}", flush=True)
if os.path.isdir(specter_dir):
    for root, _, files in os.walk(specter_dir):
        for f in files:
            p = os.path.join(root, f)
            print(f"      {os.path.getsize(p)/1e6:6.1f} MB  {p}", flush=True)

t = step("3) Cargando modelo allenai/specter… (AQUÍ es donde se cuelga la app)")
model = SentenceTransformer("allenai/specter")
print(f"    Modelo cargado  ({time.time()-t:.1f}s)", flush=True)

t = step("4) Encoding de prueba…")
vec = model.encode(["Test title. Test abstract about stroke and rTMS."])
print(f"    OK, shape={getattr(vec, 'shape', None)}  ({time.time()-t:.1f}s)", flush=True)

faulthandler.cancel_dump_traceback_later()
print("\n=== TODO OK: el modelo carga y embebe correctamente ===", flush=True)
