@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto :python_missing

if not exist ".venv\Scripts\python.exe" goto :install_environment

".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 goto :check_dependencies

echo El entorno virtual pertenece a otra instalacion de Python.
set "OLD_VENV=.venv-anterior"
if exist "%OLD_VENV%" set "OLD_VENV=.venv-anterior-%RANDOM%"
move ".venv" "%OLD_VENV%" >nul
if errorlevel 1 goto :preserve_error
echo Entorno anterior conservado en %OLD_VENV%.

:install_environment
echo Creando entorno virtual...
python -m venv .venv
if errorlevel 1 goto :error
echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

:check_dependencies
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Instalando o reparando dependencias...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo.
echo Iniciando DataLimpia Analytics...
echo Se abrira el navegador en http://localhost:8501
echo Para detener la aplicacion presiona Ctrl+C.
echo.

start "" cmd /c "timeout /t 3 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
goto :eof

:python_missing
echo.
echo No se encontro Python en el PATH. Instala Python 3.11 o superior y vuelve a intentarlo.
pause
exit /b 1

:preserve_error
echo.
echo No se pudo conservar el entorno virtual anterior. No se realizaron cambios en esa carpeta.
pause
exit /b 1

:error
echo.
echo Ocurrio un error al preparar el entorno. Revisa la instalacion de Python y tu conexion a internet.
pause
exit /b 1
