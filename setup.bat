@echo off
REM Setup para Windows. Usa uv (gestor de Python rapido).
REM Uso: doble clic en setup.bat o ejecutar desde cmd.

cd /d "%~dp0"

echo.
echo ====================================================
echo  Setup del recomendador de revistas (Windows)
echo ====================================================

REM 1) Instalar uv si no existe
where uv >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Instalando uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

REM 2) Crear venv con Python 3.12
echo Creando entorno virtual (.venv) con Python 3.12...
uv venv --python 3.12 .venv

REM 3) Instalar dependencias
echo Instalando dependencias...
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt

echo.
echo Setup completado.
echo.
echo --------------------------------------------------
echo Siguientes pasos:
echo.
echo   1) Activa el venv cada vez que abras cmd:
echo        .venv\Scripts\activate
echo.
echo   2) (Opcional) Construye el indice:
echo        python -m pipelines.run_all
echo.
echo   3) Lanza la app:
echo        streamlit run app.py
echo --------------------------------------------------
pause
